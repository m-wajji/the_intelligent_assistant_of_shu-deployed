"use client";
import Image from "next/image";
import MainLogo from "../../public/assets/main-logo.png";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const handleClick = () => {
    router.push("/home");
  };
  return (
    <main className="relative min-h-screen flex flex-col">
      <header className="flex items-center justify-between px-20 py-10">
        <div className="">
          <h1 className="text-2xl font-medium font-racing">
            Salim Habib University
          </h1>
        </div>
      </header>

      <section className="flex-grow flex flex-col items-center justify-center text-center px-4 relative">
        <div>
          <Image src={MainLogo} alt="Main Logo" />
        </div>
        <h2 className="text-2xl font-bold my-10 text-[#ada7a7cc]">
          The Intelligent Assistant Of SHU
        </h2>
        <p className="text-4xl font-semibold mb-10 w-96 font-abhaya-400">
          How may I help you today!
        </p>

        <Button
          className="w-80 h-16 bg-white px-6 py-2 rounded-3xl text-black hover:bg-[#FF0000] hover:text-white font-bold"
          onClick={handleClick}
          type="button"
        >
          Get Started
        </Button>
      </section>
    </main>
  );
}
