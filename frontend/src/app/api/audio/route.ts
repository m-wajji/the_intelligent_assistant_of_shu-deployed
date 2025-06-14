import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";

// create a session ID
function getSessionId() {
  return uuidv4();
}

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://backend:8000";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const audioFile = formData.get("audio") as Blob;
    const sessionId = getSessionId();

    if (!audioFile || !(audioFile instanceof Blob)) {
      return NextResponse.json(
        { error: "Invalid or missing audio file" },
        { status: 400 }
      );
    }

    // Fix: Create a new FormData with key "file" as expected by FastAPI
    const uploadForm = new FormData();
    uploadForm.append("file", audioFile, "recording.wav"); // use a filename if available

    const response = await fetch(`${backendUrl}/upload-audio/`, {
      method: "POST",
      body: uploadForm,
      headers: {
        session_id: sessionId,
      },
    });

    const result = await response.json();
    const filename = result.audio_response_url.split("/").pop();
    console.log(result);
    return NextResponse.json({
      transcription: result.transcription,
      response: result.answer,
      audioUrl: `/api/audio/download/${filename}`,
    });
  } catch (error) {
    console.error("Error uploading audio:", error);
    return NextResponse.json(
      { error: "Failed to upload audio" },
      { status: 500 }
    );
  }
}
