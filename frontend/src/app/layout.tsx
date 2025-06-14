import type { Metadata } from "next";
import localFont from "next/font/local";
import { Inter, Abhaya_Libre, Racing_Sans_One } from "next/font/google";
import "./globals.css";
import Background from "../../public/assets/background.png";
import Image from "next/image";

const inter = Inter({ subsets: ["latin"] });

// Import the Abhaya Libre font (SemiBold weight)
const abhayaLibre = Abhaya_Libre({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-abhaya-libre",
});

// Import Racing Sans One
const racingSansOne = Racing_Sans_One({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-racing-sans",
});

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "The Intelligent Assistant of SHU",
  description: "Agentic AI Based Query Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} ${abhayaLibre.variable} ${racingSansOne.variable} ${geistSans.variable} ${geistMono.variable} antialiased relative min-h-screen bg-black text-white`}
      >
        <div className="fixed inset-0 -z-10">
          <Image
            src={Background}
            alt="Background Image"
            fill
            style={{ objectFit: "cover" }}
          />
        </div>

        {children}
      </body>
    </html>
  );
}
