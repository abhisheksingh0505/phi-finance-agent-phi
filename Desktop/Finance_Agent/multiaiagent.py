from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckgo
from phi.agent import Agent  # Ensure you import Agent if not already done

import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckgo()],
    instruction=["Always include sources"],
    show_tool_calls=True,
)

# Financial AI Agent
financial_ai_agent = Agent(
    name="Financial AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendation=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instruction=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent Team
multi_ai_agent = Agent(
    name="Multi AI Agent",
    model=Groq(id="llama-3.1-70b-versatile"),
    team=[web_search_agent, financial_ai_agent],
    instruction=[
        "Always include sources",
        "Use tables to display the data"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Run the query
multi_ai_agent.print_response("Summarize analyst recommendation and share the latest insights.")
