import React from "react";
import { Message } from "../types";

interface Props {
  message: Message;
}

const ChatMessage: React.FC<Props> = ({ message }) => (
  <div
    className={`flex mb-4 ${
      message.sender === "user" ? "justify-end" : "justify-start"
    }`}
  >
    <div
      className={`max-w-xs md:max-w-md lg:max-w-lg p-3 rounded-lg break-words whitespace-pre-wrap ${
        message.sender === "user"
          ? "bg-blue-500 text-white"
          : "bg-gray-200 text-gray-800"
      }`}
    >
      {message.text && <p>{message.text}</p>}

      {message.pdf && (
       <a
       href={message.pdf.url}
       download
       target="_blank"
       rel="noopener noreferrer"
       className={`mt-2 inline-block underline ${
         message.sender === "user" ? "text-white" : "text-blue-600"
       } hover:text-blue-800`}
       aria-label={`Download PDF ${message.pdf.name}`}
     >
       {message.pdf.name}
     </a>
     
      )}
    </div>
  </div>
);

export default ChatMessage;
