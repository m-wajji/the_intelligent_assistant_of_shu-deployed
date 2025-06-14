export type PDFInfo = {
  name: string;
  url: string;
};

export type Message = {
  pdf?: PDFInfo; 
  id: number;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  audioUrl?: string;
  isError?: boolean;
};
