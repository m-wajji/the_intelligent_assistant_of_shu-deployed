import base64
from googleapiclient.discovery import build
from google.oauth2 import service_account
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.agent.app.core.state import State
from langgraph.types import Command
from langchain.schema import AIMessage
from langgraph.graph import END
import os
import json

# === Google Drive setup ===
def get_google_credentials():
     # Try base64 encoded version first (recommended)
    service_account_b64 = os.getenv('GOOGLE_SERVICE_ACCOUNT_B64')
    
    if service_account_b64:
        try:
            # Decode base64 and parse JSON
            service_account_json = base64.b64decode(service_account_b64).decode('utf-8')
            service_account_info = json.loads(service_account_json)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 credentials: {e}")
    else:
        # Fallback to direct JSON (with escape handling)
        service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        
        if not service_account_json:
            raise ValueError("Neither GOOGLE_SERVICE_ACCOUNT_B64 nor GOOGLE_SERVICE_ACCOUNT_JSON environment variable is set")
        
        try:
            # Clean up the JSON string - replace escaped newlines with actual newlines in private_key
            cleaned_json = service_account_json.replace('\\n', '\n')
            
            # Parse the JSON string
            service_account_info = json.loads(cleaned_json)
            
            # Ensure private_key has proper newlines
            if 'private_key' in service_account_info:
                private_key = service_account_info['private_key']
                # Replace literal \n with actual newlines if they exist
                if '\\n' in private_key:
                    service_account_info['private_key'] = private_key.replace('\\n', '\n')
                    
        except json.JSONDecodeError as e:
            # Print the problematic JSON for debugging (first 200 chars only)
            print(f"Problematic JSON (first 200 chars): {service_account_json[:200]}")
            raise ValueError(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
    
    # Create credentials from the parsed JSON
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=SCOPES
    )
    
    return credentials

def get_drive_service():
    """Get drive service, initialize only when needed"""
    try:
        credentials = get_google_credentials()
        return build("drive", "v3", credentials=credentials)
    except Exception as e:
        print(f"Failed to initialize Google Drive service: {e}")
        return None

# Don't initialize at module level - do it when needed
drive_service = None

# === PDF fetch logic ===
def fetch_pdf_from_drive(query: str) -> dict | None:
    global drive_service

    # Initialize drive service if not already done
    if drive_service is None:
        drive_service = get_drive_service()
        if drive_service is None:
            return None

    try:
        results = (
            drive_service.files()
            .list(
                q=f"fullText contains '{query}' and mimeType='application/pdf'",
                fields="files(id, name)",
                pageSize=1,
            )
            .execute()
        )

        files = results.get("files", [])
        if not files:
            return None

        file = files[0]
        return {
            "name": file["name"],
            "url": f"https://drive.google.com/uc?export=download&id={file['id']}",
        }
    except Exception as e:
        print(f"Error fetching PDF from Drive: {e}")
        return None

# === LLM setup ===
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# === Clean, strict prompt ===
prompt = PromptTemplate.from_template(
    """
If a PDF was found, respond exactly in this format:

Here is your requested PDF: {pdf_name} [Download it here]({pdf_url})

Do not add any extra explanation or text.

If no PDF was found, respond with:

Sorry, I couldn't find a PDF for that. You can check the official website page.

---

User asked: "{query}"
"""
)
chain = LLMChain(llm=llm, prompt=prompt)

# === Node function ===
def pdf_node(state: State) -> Command[str]:
    query = state.get("input", "")
    try:
        pdf = fetch_pdf_from_drive(query)
        if pdf:
            response = chain.run(query=query, pdf_name=pdf["name"], pdf_url=pdf["url"])
        else:
            response = chain.run(query=query, pdf_name="", pdf_url="")
    except Exception as e:
        response = f"An error occurred while fetching the PDF: {str(e)}"

    print("PDF Reply:", response)

    return Command(
        update={
            "messages": state.get("messages", [])
            + [AIMessage(content=response, name="pdf_tool")]
        },
        goto=END,
    )
