import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice
from pydantic import StrictStr


_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        # ⚠️ Important point: we need to provide Agent with conversation history that is related to this particular
        #    Agent, otherwise it will confuse the Agent.
        # 1. Get UMS conversation id. UMS Agent is custom implementation that is storing all the conversation on its
        #    side and without created conversation we are unable to communicate with UMS agent.
        #    The `ums_conversation_id` with be persisted in some of assistant message state (if conversation was created),
        #    additionally we will have 1-to-1 relation (one our conversation will have one conversation on the UMS agent side)
        ums_conversation_id: Optional[str] = self.__get_ums_conversation_id(request)
        # 2. If no conversation id found then create new conversation and set it to choice state as dict {_UMS_CONVERSATION_ID: {id}}
        if ums_conversation_id is None:
            ums_conversation_id = await self.__create_ums_conversation()
            stage.append_content(f" Created new UMS conversation with id: {ums_conversation_id}")
            # 3. Get last message (the last always will be the user message) and make augmentation with additional instructions
        last_user_message = request.messages[-1].content
        if additional_instructions:
            last_user_message += f"\n\nAdditional instructions for UMS Agent:\n{additional_instructions}"
        # 4. Call UMS Agent
        content = await self.__call_ums_agent(
            conversation_id=ums_conversation_id,
            user_message=last_user_message,
            stage=stage
        )
        # 5. return assistant message
        choice.set_state({_UMS_CONVERSATION_ID: ums_conversation_id})
        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
        )


    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for msg in request.messages:
            if msg.custom_content and msg.custom_content.state:
                ums_conversation_id = msg.custom_content.state.get(_UMS_CONVERSATION_ID)
                if ums_conversation_id:
                    return ums_conversation_id
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        # 1. Create async context manager with httpx.AsyncClient()
        async with httpx.AsyncClient() as client:
            # 2. Make POST request to create conversation https://github.com/khshanovskyi/ai-dial-ums-ui-agent/blob/completed/agent/app.py#L159
            # 3. Get response json and return `id` from it
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations",
                json={"title": "UMS Agent Conversation"},
                timeout=30.0
            )
            response.raise_for_status()
            conversation_data = response.json()
            return conversation_data['id']

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        # 1. Create async context manager with httpx.AsyncClient()
        async with httpx.AsyncClient() as client:
            # 2. Make POST request to chat https://github.com/khshanovskyi/ai-dial-ums-ui-agent/blob/completed/agent/app.py#L216
            #    it applies message as request body: {"message": { "role": "user","content": user_message},"stream": True}
            #    streaming must be enabled
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={
                    "message": {
                        "role": "user",
                        "content": user_message
                    },
                    "stream": True
                },
                timeout=60.0
            )
            response.raise_for_status()

            # 3. Now is the time to recall the first practice with console chat when we parsed raw streaming responses,
            #    don't worry, hopefully we made response in openai compatible (the same as in openai spec).
            #    Make async loop through `response.aiter_lines()` and:
            #       - Cut the `data: `. The streaming chunks will be returned in such format:
            #         data: {'choices': [{'delta': {'content': 'chunk 1'}}]}
            #         data: {'choices': [{'delta': {'content': 'chunk 2'}}]}
            #         data: {'choices': [{'delta': {'content': 'chunk ...'}}]}
            #         data: {'choices': [{'delta': {'content': 'chunk n'}}]}
            #         data: {'conversation_id': '{conversation_id}'}
            #         data: [DONE]
            #       - If in result you have [DONE] - that means that streaming is finished an you can break the loop
            #       - Make dict from json
            #       - Get content, accumulate it to return after and append content chunks to the stage
            content = ''
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)

                        if 'conversation_id' in data:
                            continue

                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if delta_content := delta.get('content'):
                                stage.append_content(delta_content)
                                content += delta_content
                    except json.JSONDecodeError:
                        continue

            return content
        raise NotImplementedError()