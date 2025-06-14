import { NextResponse } from "next/server";

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://backend:8000";

export async function GET() {
  // Always fetch response.mp3 since you're overwriting the same file
  const res = await fetch(
    `${backendUrl}/download-audio/response.mp3?t=${Date.now()}`
  );
  if (!res.ok) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const arrayBuffer = await res.arrayBuffer();

  return new NextResponse(Buffer.from(arrayBuffer), {
    status: 200,
    headers: {
      "Content-Type": "audio/mpeg",
      // Disable all caching to ensure fresh audio every time
      "Cache-Control": "no-cache, no-store, must-revalidate",
      Pragma: "no-cache",
      Expires: "0",
    },
  });
}
