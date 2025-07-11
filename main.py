from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.task import PipelineParams, PipelineTask
import asyncio
import sys

# from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams



from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from kokoro import KokoroTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# class TranscriptionLogger(FrameProcessor):
#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, TranscriptionFrame):
#             print(f"Transcription: {frame.text}")

async def main():
    # Configure local Ollama service
    llm = OLLamaLLMService(
        model="gemma3n:e4b",  # Must be pulled first: ollama pull llama3.1
        base_url="http://localhost:11434/v1",  # Default Ollama endpoint
        # params=OLLamaLLMService.InputParams(
        #     temperature=0.7,
        #     max_tokens=1000
        # )
    )

    # # Define function for local processing
    # weather_function = FunctionSchema(
    #     name="get_current_weather",
    #     description="Get current weather information",
    #     properties={
    #         "location": {
    #             "type": "string",
    #             "description": "City and state, e.g. San Francisco, CA"
    #         },
    #         "format": {
    #             "type": "string",
    #             "enum": ["celsius", "fahrenheit"],
    #             "description": "Temperature unit to use"
    #         }
    #     },
    #     required=["location", "format"]
    # )

    # tools = ToolsSchema(standard_tools=[weather_function])

    # Create context optimized for local model
    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful Spanish translation voicebot. Limit your responses to 2-3 sentences. Translate between English and Spanish. Respond in the same language. 
                Be concise and efficient in your responses while maintaining helpfulness. Do not output special characters such as "*" or "#" symbols as they are hard to parse via audio"""
            }
        ],
        # tools=tools
    )

    # Create context aggregators
    context_aggregator = llm.create_context_aggregator(context)

    # # Register function handler - all processing stays local
    # async def fetch_weather(params):
    #     location = params.arguments["location"]
    #     # Local weather lookup or cached data
    #     await params.result_callback({"conditions": "sunny", "temperature": "22°C"})

    # llm.register_function("get_current_weather", fetch_weather)

    # Use in pipeline - completely offline capable
    stt = WhisperSTTService()
    tts = KokoroTTSService(
            model_path="./tts-models/kokoro-v1.0.onnx",
            voices_path="./tts-models/voices.json"
        )

        
    transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                audio_out_enabled=True,
            )
        )
    pipeline = Pipeline([
        # transport.input(),
        transport.input(),
        stt,
        # stt,  # Can use local STT too
        context_aggregator.user(),
        llm,  # All inference happens locally
        tts , # Can use local TTS too
        transport.output(),
        context_aggregator.assistant()
    ])
    task = PipelineTask(pipeline,params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),)
    
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())