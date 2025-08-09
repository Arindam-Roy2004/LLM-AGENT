import streamlit as st
import requests
import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch  # Updated import - fixes deprecation warning
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="AI Assistant Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    .tool-message {
        background-color: #fff3e0;
        font-style: italic;
        font-size: 0.9rem;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .sidebar .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
    .debug-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff6b6b;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Define State
class MessagesState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage | ToolMessage], operator.add]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'workflow_initialized' not in st.session_state:
    st.session_state.workflow_initialized = False
if 'app' not in st.session_state:
    st.session_state.app = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar for API keys
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("Enter your API keys to get started:")

# API Key inputs
google_api_key = st.sidebar.text_input(
    "Google API Key", 
    type="password", 
    placeholder="Enter your Google Generative AI API key",
    help="Required for the main LLM functionality"
)

weather_api_key = st.sidebar.text_input(
    "Weather API Key", 
    type="password", 
    placeholder="Enter your WeatherAPI.com key",
    help="Required for weather information (get from weatherapi.com)"
)

tavily_api_key = st.sidebar.text_input(
    "Tavily API Key", 
    type="password", 
    placeholder="Enter your Tavily search API key",
    help="Required for web search functionality"
)

# Debug mode toggle
st.session_state.debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed error information")

# API Key Testing Section
st.sidebar.markdown("---")
st.sidebar.markdown("**🧪 API Key Testing**")

# Test Weather API
if weather_api_key:
    if st.sidebar.button("🌤️ Test Weather API"):
        with st.sidebar:
            with st.spinner("Testing Weather API..."):
                test_url = f"http://api.weatherapi.com/v1/current.json"
                test_params = {"key": weather_api_key.strip(), "q": "London"}
                try:
                    response = requests.get(test_url, params=test_params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        # Handle possible error payloads and missing keys safely
                        if isinstance(data, dict) and "error" in data:
                            msg = data["error"].get("message", "Unknown API error")
                            st.error(f"❌ Weather API Error: {msg}")
                            if st.session_state.debug_mode:
                                st.write(data)
                        else:
                            current = data.get("current", {}) if isinstance(data, dict) else {}
                            location = data.get("location", {}) if isinstance(data, dict) else {}
                            temp_c = current.get("temp_c")
                            loc_name = location.get("name")
                            if temp_c is not None and loc_name:
                                st.success("✅ Weather API key is working!")
                                st.write(f"🌡️ Test: {temp_c}°C in {loc_name}")
                            else:
                                st.warning("⚠️ Unexpected response from Weather API.")
                                if st.session_state.debug_mode:
                                    st.write(data)
                    elif response.status_code == 401:
                        st.error("❌ Invalid API key")
                    elif response.status_code == 403:
                        st.error("❌ API key quota exceeded")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        if st.session_state.debug_mode:
                            st.write(response.text)
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

# Test Tavily API
if tavily_api_key:
    if st.sidebar.button("🔍 Test Search API"):
        with st.sidebar:
            with st.spinner("Testing Tavily API..."):
                try:
                    test_search = TavilySearch(tavily_api_key=tavily_api_key.strip(), max_results=1)
                    response = test_search.invoke({"query": "test search"})
                    if response:
                        st.success("✅ Tavily API key is working!")
                        st.write("🔍 Test search completed successfully")
                    else:
                        st.error("❌ Tavily API test failed")
                except Exception as e:
                    st.error(f"❌ Tavily API Error: {str(e)}")

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose LLM Model",
    ["gemini-2.5-pro"],  # Only Gemini since OpenAI isn't fully implemented
    help="Select the language model to use"
)

# Temperature setting
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Controls randomness in responses"
)

def create_tools(weather_key, tavily_key, debug_mode=False):
    """Create tools with the provided API keys"""
    
    @tool
    def get_weather(city: str):
        """Use this tool ONLY to find the current weather for a specific city."""
        if not weather_key:
            return "❌ Error: Weather API key is not configured."
        
        # Clean the city input
        city = city.strip()
        if not city:
            return "❌ Error: City name cannot be empty."
        
        url = "http://api.weatherapi.com/v1/current.json"
        params = {
            "key": weather_key.strip(),
            "q": city,
            "aqi": "no"
        }
        
        headers = {
            "User-Agent": "Streamlit-Weather-App/1.0"
        }
        
        if debug_mode:
            st.write("🔍 **DEBUG INFO:**")
            st.write(f"- API URL: {url}")
            st.write(f"- City query: '{city}'")
            st.write(f"- API key length: {len(weather_key)} chars")
            st.write(f"- API key starts with: {weather_key[:8]}...")
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if debug_mode:
                st.write(f"- Response status: {response.status_code}")
                st.write(f"- Response headers: {dict(response.headers)}")
            
            # Check for specific HTTP errors
            if response.status_code == 401:
                return "❌ **API Key Error**: Invalid WeatherAPI.com key. Please verify your API key from weatherapi.com dashboard."
            elif response.status_code == 400:
                return f"❌ **Location Error**: '{city}' not found. Try a more specific location like 'Silchar, India'."
            elif response.status_code == 403:
                return "❌ **Quota Error**: API key quota exceeded or access denied. Check your weatherapi.com account limits."
            elif response.status_code == 429:
                return "❌ **Rate Limit**: Too many requests. Please wait a moment before trying again."
            
            response.raise_for_status()
            data = response.json()
            
            if debug_mode:
                st.write(f"- Response data keys: {list(data.keys())}")
            
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown API error")
                return f"❌ **Weather API Error**: {error_msg}"
            
            # Extract weather data with error checking
            location = data.get("location", {})
            current = data.get("current", {})
            
            if not location or not current:
                return "❌ **Data Error**: Incomplete weather data received from API."
            
            location_name = location.get("name", "Unknown")
            region = location.get("region", "")
            country = location.get("country", "Unknown")
            temp_c = current.get("temp_c", "N/A")
            condition = current.get("condition", {}).get("text", "Unknown")
            feelslike_c = current.get("feelslike_c", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_kph = current.get("wind_kph", "N/A")
            
            # Format location string
            location_str = f"{location_name}"
            if region and region != location_name:
                location_str += f", {region}"
            location_str += f", {country}"
            
            return (f"🌤️ **Weather in {location_str}**\n\n"
                   f"🌡️ **Temperature**: {temp_c}°C (feels like {feelslike_c}°C)\n"
                   f"☁️ **Condition**: {condition}\n"
                   f"💧 **Humidity**: {humidity}%\n"
                   f"💨 **Wind Speed**: {wind_kph} km/h")
            
        except requests.exceptions.Timeout:
            return "⏰ **Timeout Error**: Request timed out. Please check your internet connection and try again."
        except requests.exceptions.ConnectionError:
            return "🌐 **Connection Error**: Cannot connect to weather service. Please check your internet connection."
        except requests.exceptions.RequestException as e:
            return f"🔗 **Network Error**: {str(e)}"
        except Exception as e:
            error_msg = f"❌ **Unexpected Error**: {str(e)}"
            if debug_mode:
                import traceback
                error_msg += f"\n\n**Traceback**:\n``````"
            return error_msg
    
    @tool
    def get_stock_price(ticker: str):
        """Fetch the previous closing price of asked stocks.
        Args:
            ticker (str): The stock ticker symbol.
        Returns:
            str: The previous closing price of the stock with currency symbol or an error message.
        """
        try:
            ticker = ticker.strip().upper()
            if debug_mode:
                st.write(f"🔍 **DEBUG**: Fetching stock data for {ticker}")
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                return f"❌ **Stock Error**: No data found for ticker '{ticker}'. Please verify the symbol."
            
            last_price = hist['Close'].iloc[-1]
            date = hist.index[-1].strftime('%Y-%m-%d')
            
            # Get additional info
            try:
                info = stock.info
                company_name = info.get('longName', ticker)
            except:
                company_name = ticker
            
            return (f"📈 **{company_name} ({ticker})**\n\n"
                   f"💰 **Closing Price**: ${last_price:.2f}\n"
                   f"📅 **Date**: {date}")
            
        except Exception as e:
            error_msg = f"❌ **Stock API Error**: Failed to fetch data for '{ticker}': {str(e)}"
            if debug_mode:
                import traceback
                error_msg += f"\n\n**Traceback**:\n``````"
            return error_msg
    
    # Replace Tavily direct tool with a wrapper named exactly "search_tool"
    if tavily_key:
        @tool("search_tool")
        def search_tool(query: str) -> str:
            """Use this for real-world, factual, or up-to-date information (events, dates, news, stats)."""
            try:
                ts = TavilySearch(tavily_api_key=tavily_key.strip(), max_results=5)
                result = ts.invoke({"query": query})
                
                if debug_mode:
                    st.write("🔍 Tavily raw result:")
                    try:
                        st.write(result if isinstance(result, (dict, list)) else str(result)[:1000])
                    except Exception:
                        pass
                
                # Normalize to a list of items
                items = None
                if isinstance(result, list):
                    items = result
                elif isinstance(result, dict):
                    if isinstance(result.get("results"), list):
                        items = result["results"]
                
                if items:
                    lines = []
                    for i, item in enumerate(items[:3], 1):
                        title = (
                            item.get("title") if isinstance(item, dict) else None
                        ) or "Result"
                        url = (
                            item.get("url") if isinstance(item, dict) else None
                        ) or ""
                        snippet = (
                            item.get("content") if isinstance(item, dict) else None
                        ) or (item.get("snippet") if isinstance(item, dict) else "")
                        part = f"{i}. {title}\n{url}\n{snippet}".strip()
                        lines.append(part)
                    return "🔎 Top results:\n\n" + "\n\n".join(lines)
                
                # Fallback to string representation
                return str(result)
            except Exception as e:
                return f"❌ **Search Error**: {str(e)}"
    else:
        @tool("search_tool")
        def search_tool(query: str) -> str:
            """Dummy search tool when API key is not provided"""
            return "❌ **Search Unavailable**: Please provide a Tavily API key to enable web search functionality."
    
    return [get_weather, get_stock_price, search_tool]

def initialize_workflow(google_key, weather_key, tavily_key, model, temp, debug_mode=False):
    """Initialize the LangGraph workflow"""
    
    # Set environment variables
    os.environ["GOOGLE_API_KEY"] = google_key
    if weather_key:
        os.environ["WEATHERAPI_API_KEY"] = weather_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    
    # Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=temp)
    except Exception as e:
        st.error(f"❌ **LLM Initialization Error**: {str(e)}")
        raise e
    
    # Create tools
    tools = create_tools(weather_key, tavily_key, debug_mode)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    # Define agent nodes
    def general_agent_node(state: MessagesState):
        """Handles general conversation without tools."""
        try:
            response = llm.invoke(state['messages'])
            return {"messages": [response]}
        except Exception as e:
            if debug_mode:
                st.write(f"🔧 General agent error: {str(e)}")
            # Return a simple error message
            error_response = AIMessage(content=f"❌ I encountered an error: {str(e)}")
            return {"messages": [error_response]}
    
    def tool_agent_node(state: MessagesState):
        """Decides if a tool is needed and generates the tool call."""
        try:
            response = llm_with_tools.invoke(state['messages'])
            return {"messages": [response]}
        except Exception as e:
            if debug_mode:
                st.write(f"🔧 Tool agent error: {str(e)}, falling back to general agent")
            # Fallback to general agent if tool agent fails
            try:
                fallback_response = llm.invoke(state['messages'])
                return {"messages": [fallback_response]}
            except Exception as fallback_error:
                error_response = AIMessage(content=f"❌ I encountered an error: {str(fallback_error)}")
                return {"messages": [error_response]}
    
    # Define routers
    def route_message(state: MessagesState):
        """Routes the query to the appropriate agent based on keywords."""
        try:
            message = state['messages'][-1]
            content = message.content.lower()
            tool_keywords = [
                "weather", "temperature", "climate", "forecast",
                "who is", "what is", "when", "when was", "find", "search", 
                "date", "day", "score", "price", "won", "championship", 
                "cricket", "stock", "share", "match", "current score", 
                "latest news", "name", "politics", "president", "tariff", 
                "recently", "today", "yesterday", "india", "country", 
                "tier", "college", "university", "ranking", "level", "status",
                "rahul gandhi", "eci", "election commission","Nit Silchar","Nit", "Silchar", "Assam", "India","todays date", "current date", "today's date", "current time", "time in India"
            ]
            if any(keyword in content for keyword in tool_keywords):
                return "tool_path"
            else:
                return "general_path"
        except Exception as e:
            if debug_mode:
                st.write(f"🔧 Router error: {str(e)}, defaulting to general path")
            return "general_path"
    
    def tool_call_router(state: MessagesState):
        """Routes to the tool executor or ends the conversation."""
        try:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls and len(last_message.tool_calls) > 0:
                return "call_tool"
            else:
                return "end_conversation"
        except Exception as e:
            if debug_mode:
                st.write(f"🔧 Tool router error: {str(e)}, ending conversation")
            return "end_conversation"
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("tool_agent", tool_agent_node)
    workflow.add_node("tool_executor", tool_node)
    
    workflow.set_conditional_entry_point(route_message, {"general_path": "general_agent", "tool_path": "tool_agent"})
    workflow.add_edge("general_agent", END)
    workflow.add_conditional_edges("tool_agent", tool_call_router, {"call_tool": "tool_executor", "end_conversation": END})
    workflow.add_edge("tool_executor", "tool_agent")
    
    return workflow.compile()

# System prompt
system_prompt_final = (
    "### Persona\n"
    "You are a friendly, highly capable, and expert digital assistant. "
    "Your primary goal is to provide accurate and helpful information to the user in a conversational manner.\n\n"
    "### Core Directives\n"
    "1.  **Primary Tool**: Your main tool for answering questions is the `search_tool`. "
    "You should default to using it for any query that requires factual, real-world, or up-to-date information.\n"
    "2.  **Timezone Context**: You MUST operate as if you are in the Indian Standard Time (IST) zone. "
    "All time-related statements and queries should be interpreted and answered relative to IST.\n\n"
    "### Tool Usage Protocol\n"
    "1.  **Weather Tool**: If a user's query is *specifically* and *only* about the current weather or temperature in a city, "
    "you MUST use the `get_weather` tool.\n"
    "2.  **Search Tool**: For ANY other question that requires external information—including but not limited to events, dates, facts, "
    "3.  **Stock Price Tool**: If a user's query is specifically about the previous closing price of a stock, "
    "statistics, news, or general knowledge—you MUST use the `search_tool`.\n"
    "4.  **Execution**: When you decide to use a tool, generate the tool call directly and immediately. "
    "Do not output the code for the tool call or ask for permission to use it.\n"
    "5.  **Information Synthesis**: After receiving information from a tool, you MUST synthesize it into a comprehensive, "
    "user-friendly answer. Do not just state the raw tool output. Combine the tool's data with the user's original question to form a complete response.\n\n"
    "### Response Generation\n"
    "1.  **Clarity and Formatting**: Present your final answer clearly. Use markdown (like bullet points, bolding) to structure the information and make it easy to read.\n"
    "2.  **Engaging Tone**: Be friendly and conversational. Using relevant emojis (like 🏏 for cricket) is encouraged to make your responses more engaging.\n"
    "3.  **Completeness**: Ensure you provide a complete, final answer after the tool use cycle is complete.\n"
    "4.  **Readability**: Format your responses to be easily readable on a variety of screen sizes. Use line breaks to prevent long lines of text that would require horizontal scrolling."
)

# Main app
st.title("🤖 AI Assistant Chat")
st.markdown("Your intelligent assistant powered by LangGraph and multiple AI tools!")

# Check if minimum required keys are provided
if not google_api_key:
    st.warning("⚠️ Please provide at least the Google API key to start chatting.")
    st.info("💡 **Getting started:**\n- Google API Key is required for basic functionality\n- Weather API Key enables weather queries (get from weatherapi.com)\n- Tavily API Key enables web search capabilities")
else:
    # Initialize workflow if not already done or if keys changed
    if (not st.session_state.workflow_initialized or 
        st.session_state.get('last_google_key') != google_api_key or
        st.session_state.get('last_weather_key') != weather_api_key or
        st.session_state.get('last_tavily_key') != tavily_api_key or
        st.session_state.get('last_debug_mode') != st.session_state.debug_mode):
        
        try:
            with st.spinner("🔄 Initializing AI assistant..."):
                st.session_state.app = initialize_workflow(
                    google_api_key, weather_api_key, tavily_api_key, 
                    model_choice, temperature, st.session_state.debug_mode
                )
                st.session_state.workflow_initialized = True
                st.session_state.last_google_key = google_api_key
                st.session_state.last_weather_key = weather_api_key
                st.session_state.last_tavily_key = tavily_api_key
                st.session_state.last_debug_mode = st.session_state.debug_mode
            st.success("✅ AI assistant initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing assistant: {str(e)}")
            if st.session_state.debug_mode:
                import traceback
                st.code(traceback.format_exc())
            st.stop()

    # Chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                st.markdown(f"*Tool: {message['content']}*")

    # Chat input
    if prompt := st.chat_input("Ask me anything! I can help with weather, stocks, search, and general questions."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("🤔 Thinking..."):
                    inputs = {
                        "messages": [
                            SystemMessage(content=system_prompt_final),
                            HumanMessage(content=prompt)
                        ]
                    }
                    
                    response_content = ""
                    tool_calls_made = []
                    
                    # Stream the agent's process
                    for event in st.session_state.app.stream(inputs, {"recursion_limit": 10}):
                        for value in event.values():
                            if 'messages' in value:
                                message = value['messages'][-1]
                                
                                if isinstance(message, AIMessage):
                                    if message.tool_calls:
                                        tool_name = message.tool_calls[0]['name']
                                        tool_args = message.tool_calls[0]['args']
                                        tool_calls_made.append(f"🔧 Using tool: **{tool_name}** with arguments: {tool_args}")
                                    else:
                                        response_content = message.content
                                        break
                                
                                elif isinstance(message, ToolMessage):
                                    tool_calls_made.append(f"🔍 Tool **{message.name}** completed")
                    
                    # Display tool calls if any
                    if tool_calls_made:
                        with st.expander("🔧 Tool Usage Details", expanded=st.session_state.debug_mode):
                            for tool_call in tool_calls_made:
                                st.markdown(tool_call)
                    
                    # Display final response
                    if response_content:
                        message_placeholder.markdown(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        error_msg = "❌ I apologize, but I couldn't generate a response. Please try again."
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
            except Exception as e:
                error_msg = f"❌ **Error occurred**: {str(e)}"
                if st.session_state.debug_mode:
                    import traceback
                    error_msg += f"\n\n**Debug Traceback**:\n``````"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button and footer
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**✨ Features Available:**")
st.sidebar.markdown("- 🌤️ Weather information")
st.sidebar.markdown("- 📈 Stock prices")
st.sidebar.markdown("- 🔍 Web search")
st.sidebar.markdown("- 💬 General conversation")
st.sidebar.markdown("- 🐛 Debug mode for troubleshooting")

if st.session_state.debug_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("🐛 **Debug Mode Active**")
    st.sidebar.markdown("Detailed error information will be shown")
