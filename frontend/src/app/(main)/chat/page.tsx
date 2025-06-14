"use client";

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Send, Mic } from "lucide-react";
import { motion } from "framer-motion";
import ChatMessage from "../../../components/ChatMessage";
import AudioMessage from "../../../components/AudioMessage";
import { Message } from "../../../types";

function parsePdfFromResponse(responseText: string) {
  // Regex to find markdown link: [link text](url)
  const mdLinkRegex =
    /\[([^\]]+)\]\((https:\/\/drive\.google\.com\/uc\?export=download&id=[^)]+)\)/;

  const match = responseText.match(mdLinkRegex);

  if (!match) return { cleanText: responseText, pdf: undefined };

  const [, pdfName, pdfUrl] = match;

  // Remove the whole markdown link line from the response text
  // Also remove any empty lines around it
  const cleanText = responseText
    .replace(mdLinkRegex, "")
    .replace(/\n{2,}/g, "\n") // collapse multiple new lines to one
    .trim();

  return {
    cleanText,
    pdf: { name: pdfName, url: pdfUrl },
  };
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm your AI assistant. How can I help you today?",
      sender: "ai",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [recording, setRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentlyPlayingId, setCurrentlyPlayingId] = useState<number | null>(
    null
  );
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const startRecording = async () => {
    if (isProcessing) return;

    // Stop any currently playing audio
    setCurrentlyPlayingId(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunks.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        await sendAudioMessage(audioBlob);
        // Clean up the stream
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const sendAudioMessage = async (audioBlob: Blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");

    try {
      const { data } = await axios.post("/api/audio", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Check if there's an error (like unsupported language)
      if (data.error) {
        const errorMsg: Message = {
          id: messages.length + 1,
          text:
            data.message || "An error occurred while processing your audio.",
          sender: "ai",
          timestamp: new Date().toISOString(),
          isError: true,
        };

        // If it's an unsupported language, add additional context
        if (data.error === "unsupported_language" && data.detected_language) {
          errorMsg.text += ` (Detected language: ${data.detected_language})`;
        }

        setMessages((prev) => [...prev, errorMsg]);
        return;
      }

      // If transcription is available, add user message
      if (data.transcription) {
        const userMsg: Message = {
          id: messages.length + 1,
          text: data.transcription,
          sender: "user",
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMsg]);
      }

      // Add AI response
      const aiMsg: Message = {
        id: messages.length + (data.transcription ? 2 : 1),
        text: data.response || data.answer,
        sender: "ai",
        timestamp: new Date().toISOString(),
        audioUrl: data.audioUrl || data.audio_response_url,
      };

      setMessages((prev) => [...prev, aiMsg]);

      // Set the AI message as currently playing if audio is available
      if (aiMsg.audioUrl) {
        setCurrentlyPlayingId(aiMsg.id);
      }
    } catch (error) {
      console.error("Error sending audio message:", error);

      // Add error message to chat
      const errorMsg: Message = {
        id: messages.length + 1,
        text: "Sorry, I currently handle English and Urdu only.",
        sender: "ai",
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async () => {
    if (isProcessing || !inputMessage.trim()) return;

    setIsProcessing(true);
    const newUserMessage: Message = {
      id: messages.length + 1,
      text: inputMessage,
      sender: "user",
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage("");

    try {
      const response = await axios.post("/api/chat", {
        message: newUserMessage.text,
      });

      // Check if there's an error (like unsupported language)
      if (response.data.error) {
        const errorMsg: Message = {
          id: newUserMessage.id + 1,
          text:
            response.data.message ||
            "An error occurred while processing your message.",
          sender: "ai",
          timestamp: new Date().toISOString(),
          isError: true,
        };

        // If it's an unsupported language, add additional context
        if (
          response.data.error === "unsupported_language" &&
          response.data.detected_language
        ) {
          errorMsg.text += ` (Detected language: ${response.data.detected_language})`;
        }

        setMessages((prev) => [...prev, errorMsg]);
        return;
      }

      // Parse PDF from response if present
      const { cleanText, pdf } = parsePdfFromResponse(
        response.data.response || response.data.answer
      );

      const aiMessage: Message = {
        id: newUserMessage.id + 1,
        text: cleanText || "Please see the attached PDF.",
        sender: "ai",
        timestamp: new Date().toISOString(),
        audioUrl: response.data.audioUrl,
        pdf: pdf || response.data.pdf, // Use parsed PDF or backend PDF info
      };

      setMessages((prev) => [...prev, aiMessage]);

      if (response.data.audioUrl) {
        setCurrentlyPlayingId(aiMessage.id);
      }
    } catch (error) {
      console.error("Error fetching response:", error);

      // Add error message to chat
      const errorMsg: Message = {
        id: newUserMessage.id + 1,
        text: "Sorry, I currently handle English and Urdu only.",
        sender: "ai",
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAudioEnded = () => {
    setCurrentlyPlayingId(null);
  };

  return (
    <main className="h-screen flex flex-col">
      <header className="pt-10 pb-4 flex items-center justify-center">
        <h1 className="text-2xl font-bold text-[#ada7a7cc]">Chat with me</h1>
      </header>
      <section className="flex-1 overflow-y-auto p-4 h-[calc(100vh-120px)] pb-14">
        <div className="max-w-4xl mx-auto">
          {messages.map((msg) =>
            msg.audioUrl ? (
              <div key={msg.id}>
                <ChatMessage message={msg} />
                <AudioMessage
                  src={msg.audioUrl}
                  onEnded={() => handleAudioEnded()}
                  autoplay={
                    currentlyPlayingId === msg.id && !recording && !isProcessing
                  }
                />
              </div>
            ) : (
              <ChatMessage key={msg.id} message={msg} />
            )
          )}
          <div ref={messagesEndRef} />
        </div>
      </section>
      <div className="absolute bottom-0 left-0 w-full p-4 pt-10">
        <div className="max-w-4xl mx-auto flex">
          <input
            type="text"
            className="w-full px-4 py-2 border rounded-l-md text-black focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Type your message..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) =>
              !isProcessing && e.key === "Enter" && handleSendMessage()
            }
            disabled={isProcessing}
          />
          {inputMessage ? (
            <button
              className={`px-4 py-2 bg-blue-500 text-white rounded-r-md hover:bg-blue-600 focus:outline-none ${
                isProcessing ? "opacity-50 cursor-not-allowed" : ""
              }`}
              onClick={handleSendMessage}
              disabled={isProcessing}
            >
              <Send size={18} />
            </button>
          ) : (
            <motion.button
              className={`px-4 py-2 rounded-r-md text-white transition-all duration-300 relative flex items-center justify-center ${
                recording ? "bg-red-600" : "bg-gray-500 hover:bg-gray-600"
              } ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
              onClick={recording ? stopRecording : startRecording}
              animate={recording ? { scale: [1, 1.1, 1] } : { scale: 1 }}
              transition={{ duration: 0.8, repeat: recording ? Infinity : 0 }}
              whileTap={{ scale: 0.9 }}
              disabled={isProcessing}
            >
              <Mic size={18} />
              {recording && (
                <motion.span
                  className="absolute top-1 right-1 w-2 h-2 bg-red-300 rounded-full"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
            </motion.button>
          )}
        </div>
      </div>
    </main>
  );
};

export default Chat;
