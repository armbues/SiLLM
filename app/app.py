import os
import pathlib
import re
import json

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput, Switch

import mcp

import sillm
import sillm.utils as utils

log_level = 10
logger = utils.init_logger(log_level, "%(asctime)s - %(message)s", add_stdout=False)

# Initialize model paths
MODEL_DIR = os.environ.get("SILLM_MODEL_DIR")
if MODEL_DIR is None:
    raise ValueError("Please set the environment variable SILLM_MODEL_DIR to the directory containing the models.")
MODEL_PATHS = {model_path.name: model_path for model_path in pathlib.Path(MODEL_DIR).iterdir() if model_path.is_dir() or model_path.suffix == ".gguf"}

current_model = None
current_model_name = ""

@cl.cache
def load_model(model_name: str):
    return sillm.load(MODEL_PATHS[model_name]), model_name

@cl.on_chat_start
async def on_chat_start():
    model_names = list(MODEL_PATHS.keys())
    model_names.sort()

    templates = ["[default]"] + [template.removesuffix(".jinja") for template in sillm.Template.list_templates()]
    templates.sort()

    settings = [
        Select(id="Model", label="Model", values=model_names, initial_index=0),
        Select(id="Template", label="Chat Template", values=templates, initial_index=0),
        Slider(id="Tokens", label="Max. Tokens", initial=4096, min=1024, max=32768, step=1024),
        Switch(id="Defaults", label="Default Generation Settings (generation_config.json)", initial=False),
        Slider(id="Temperature", label="Model Temperature", initial=0.7, min=0, max=2, step=0.1),
        Slider(id="TopK", label="Top-K", initial=0, min=0, max=100, step=1),
        Slider(id="TopP", label="Top-P", initial=1.0, min=0.1, max=1.0, step=0.01),
        Slider(id="MinP", label="Min-P", initial=0.0, min=0.0, max=1.0, step=0.01),
        Slider(id="Penalty", label="Repetition Penalty", initial=1.0, min=0.1, max=3.0, step=0.1),
        Slider(id="Window", label="Repetition Window", initial=50, min=10, max=200, step=10),
        TextInput(id="Seed", label="Seed", initial="-1"),
    ]

    await cl.ChatSettings(settings).send()

    generate_args = {
        "temperature": 0.7,
        "max_tokens": 2048,
        "flush": 5
    }
    cl.user_session.set("generate_args", generate_args)

    commands = [
        {
            "id": "system",
            "icon": "scroll-text",
            "description": "Set system prompt",
        }
    ]
    await cl.context.emitter.set_commands(commands)

    cl.user_session.set("conversation", None)

@cl.on_settings_update
async def on_settings_update(settings):
    model_name = cl.user_session.get("model_name")
    if model_name != settings["Model"]:
        cl.user_session.set("model_name", settings["Model"])

        # TODO implement evaluate on model change
        cl.user_session.set("conversation", None)
        cl.user_session.set("cache", None)

    cl.user_session.set("template_name", settings["Template"])
    cl.user_session.set("defaults", settings["Defaults"])

    generate_args = {
        "temperature": settings["Temperature"],
        "max_tokens": settings["Tokens"]
    }
    if settings["TopK"] > 0:
        generate_args["top_k"] = settings["TopK"]
    if settings["TopP"] < 1.0:
        generate_args["top_p"] = settings["TopP"]
    if settings["MinP"] > 0.0:
        generate_args["min_p"] = settings["MinP"]
    if settings["Penalty"] != 1.0:
        generate_args["repetition_penalty"] = settings["Penalty"]
        generate_args["repetition_window"] = settings["Window"]
    cl.user_session.set("generate_args", generate_args)

    seed = int(settings["Seed"])
    if seed >= 0:
        utils.seed(seed)

@cl.on_message
async def on_message(message: cl.Message):
    global current_model
    global current_model_name
    
    # Execute commands
    if message.command is not None:
        command = message.command

        if command == "system":
            cl.user_session.set("system_prompt", message.content)
            await cl.context.emitter.send_toast(f"System prompt defined.", type="info")
            
            return

    # Initialize model
    model_name = cl.user_session.get("model_name")
    
    # Check if no model is selected
    if model_name is None:
        await cl.context.emitter.send_toast("Please select a model.", type="error")
        return
    
    # Load model if not already loaded
    if current_model_name == model_name:
        model = current_model
    else:
        await cl.context.emitter.send_toast(f"Loading model {model_name}", type="info")

        current_model, current_model_name = load_model(model_name)
        model = current_model

        await cl.context.emitter.send_toast(f"Finished loading {model_name}", type="success")

    # Fetch MCP tools
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tools = [tool for connection_tools in mcp_tools.values() for tool in connection_tools]

    # Initialize conversation & templates
    conversation = cl.user_session.get("conversation")
    system_prompt = cl.user_session.get("system_prompt")
    if conversation is None:
        template_name = cl.user_session.get("template_name")
        if template_name == "[default]":
            template_name = None

        template = sillm.init_template(model.tokenizer, model.args, template_name)
        conversation = sillm.Conversation(template, system_prompt=system_prompt, tools=tools)
        cl.user_session.set("conversation", conversation)
    else:
        conversation.tools = tools

    # Initialize cache
    cache = cl.user_session.get("cache")
    if cache is None:
        cache = model.init_kv_cache()
        cl.user_session.set("cache", cache)

    # Add user message to conversation and get prompt string
    request = conversation.add_user(message.content)
    
    # Get model generation arguments
    generate_args = cl.user_session.get("generate_args")

    if cl.user_session.get("defaults"):
        # Use default generation config
        if model.args.generation_config is not None:
            generate_args.update(model.args.generation_config)

    logger.debug(f"Generating {generate_args['max_tokens']} tokens with temperature {generate_args['temperature']}")

    while request is not None:
        msg = cl.Message(author=model_name, content="")
        step = None
        mode = "response"
        
        # Generate response
        response = ""
        for s, _ in model.generate(request, cache=cache, **generate_args):
            response += s

            tag_re = re.compile(r'(</?think>|</?tool_call>)')
            parts = tag_re.split(s)

            new_response = ""
            new_think = ""

            for part in parts:
                if tag_re.fullmatch(part):
                    if part == "<think>":
                        mode = "thinking"
                    elif part == "</think>":
                        mode = "response"
                    elif part == "<tool_call>":
                        mode = "tool_call"
                    elif part == "</tool_call>":
                        mode = "response"
                else:
                    if mode == "response":
                        new_response += part
                    elif mode == "thinking":
                        new_think += part

            if new_response.strip():
                await msg.stream_token(new_response)
            if new_think.strip():
                if step is None:
                    step = cl.Step(name="Reasoning", parent_id=msg.parent_id)
                await step.stream_token(new_think)

        if msg.content != "":
            await msg.send()
        if step is not None:
            await step.send()

        # Add response to conversation
        conversation.add_assistant(response)

        # Handle tool calls
        request = None
        if '<tool_call>' in response and '</tool_call>' in response:
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)

            results = []
            for tool_call in tool_calls:
                tool_call = json.loads(tool_call)
                conversation.add_message(tool_call, role="tool")

                result = await call_tool(tool_call["name"], tool_call["arguments"])
                results.append(result)
            
            request = conversation.add_tool_calls(results)

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: mcp.ClientSession):
    # Fetch tools from MCP connection
    mcp_tools = cl.user_session.get("mcp_tools", {})
    result = await session.list_tools()
    mcp_tools[connection.name] = [t.model_dump() for t in result.tools]
    cl.user_session.set("mcp_tools", mcp_tools)

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: mcp.ClientSession):
    # Clean up tools after MCP disconnect
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

@cl.step(type="tool", name="Tool Call")
async def call_tool(name, arguments):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None
    for connection_name, tools in mcp_tools.items():
        for tool in tools:
            if tool["name"] == name:
                mcp_name = connection_name
                break

    # Handle missing tool
    if mcp_name is None:
        return f"Error: tool {name} not found."
    
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    tool_response = await mcp_session.call_tool(name, arguments)
    if tool_response.isError is True:
        result = "Error:\n"
    else:
        result = ""

    for content in tool_response.content:
        if content.type == "text":
            result += content.text

    try:
        json.loads(result)
    except:
        pass
    finally:
        cl.context.current_step.language = "json"
        await cl.context.current_step.update()

    return result

async def update_commands():
    commands = []

    mcp_system_prompts = cl.user_session.get("mcp_system_prompts", {})
    for connection_name in mcp_system_prompts.keys():
        command = {
            "id": f"{connection_name} System Prompt",
            "icon": "scroll-text"
        }
        commands.append(command)

    if len(commands) > 0:
        await cl.context.emitter.set_commands(commands)