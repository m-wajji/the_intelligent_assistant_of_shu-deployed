import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";

// create a session ID
function getSessionId() {
  return uuidv4();
}

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://backend:8000";

export async function POST(req: Request) {
  try {
    const { message } = await req.json();
    const sessionId = getSessionId();

    console.log(`Using session ID: ${sessionId}`);

    const response = await fetch(`${backendUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        session_id: sessionId,
      },
      body: JSON.stringify({ question: message }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error(`Backend error: ${response.status}`, errorData);

      if (response.status === 403) {
        return NextResponse.json(
          { error: "Query limit exceeded. Starting a new session." },
          { status: 403 }
        );
      }

      throw new Error(`Backend error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json({ response: data.answer });
  } catch (error) {
    console.error("Error communicating with backend:", error);
    return NextResponse.json({ error: "Chat failed" }, { status: 500 });
  }
}
