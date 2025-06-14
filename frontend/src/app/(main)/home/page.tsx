"use client";
import React from "react";
import { Button } from "@/components/ui/button";
import { TbArrowUpRight, TbBrandWechat } from "react-icons/tb";
import { MdKeyboardArrowRight } from "react-icons/md";
import { useRouter } from "next/navigation";

const Home = () => {
  const router = useRouter();

  const handleClick = (path: string) => {
    router.push(path);
  };

  const faqs = [
    {
      faqNo: 1,
      question: "Where is SHU located?",
    },
    {
      faqNo: 2,
      question: "Is there any scholarship available?",
    },
    {
      faqNo: 3,
      question: "Can I apply for admission online?",
    },
  ];
  return (
    <main className="flex flex-col ">
      <header className="p-16">
        <h1 className="text-2xl font-bold text-[#ada7a7cc]">
          The Intelligent Assistant of SHU
        </h1>
      </header>
      <section className="flex flex-col items-center">
        <div className="w-[220px]">
          <h1 className="font-medium text-3xl leading-[45px]">
            How may I help you today?
          </h1>
        </div>

        <div className="flex my-14 items-center justify-center">
          {/* <Button
            className="w-[185px] h-[150px] bg-[#FF0000] rounded-3xl flex flex-col  "
            onClick={() => {
              handleClick("/talk");
            }}
            type="button"
          >
            <div className="flex justify-between items-center -mt-5 w-full">
              <div className="w-11 h-11 rounded-full bg-[#FFFFFFC4] flex items-center justify-center">
                <RiSpeakLine
                  color="black"
                  // className="text-black"
                  style={{ width: "30px", height: "30px" }}
                />
              </div>
              <TbArrowUpRight
                // className=""
                color="black"
                style={{ width: "30px", height: "30px" }}
              />
            </div>
            <div className="items-start flex mt-6">
              <h1 className=" text-[27px]">Talk with me</h1>
            </div>
          </Button> */}
          <Button
            className="w-[280px] h-[150px] bg-[#FFFFFFD6] rounded-3xl flex flex-col"
            onClick={() => {
              handleClick("/chat");
            }}
            type="button"
          >
            <div className="flex justify-between items-center -mt-3 w-full">
              <div className="w-12 h-12 rounded-full bg-[#00000033] flex items-center justify-center">
                <TbBrandWechat
                  color="black"
                  // className="text-black"
                  style={{ width: "45px", height: "45px" }}
                />
              </div>
              <TbArrowUpRight
                // className=""
                color="black"
                style={{ width: "45px", height: "45px" }}
              />
            </div>
            <div className="items-start flex mt-6">
              <h1 className=" text-2xl text-black">Start the Conversation</h1>
            </div>
          </Button>
        </div>

        <div className="flex justify-between items-center w-1/3 mt-12">
          <h2 className="text-base font-bold text-[#FFFFFF7A]">
            Frequently Asked Questions
          </h2>
          <Button
            className="text-base font-bold text-[#FFFFFF7A] bg-transparent"
            onClick={() => {
              handleClick("/faq");
            }}
            type="submit"
          >
            See all
          </Button>
        </div>

        <div className="flex flex-col gap-2 mt-3 w-1/3">
          {faqs.map((faq) => {
            return (
              <Button
                key={faq.faqNo}
                className="flex items-center justify-between px-4 py-5 bg-[#FF000070] rounded-3xl border border-[#FFFFFF1A] h-[80px]"
                type="button"
                onClick={() => {
                  handleClick("/faq");
                }}
              >
                <div className="w-11 h-11 rounded-full bg-[#FFFFFFB0] flex items-center justify-center">
                  <TbBrandWechat
                    color="black"
                    // className="text-black"
                    style={{ width: "30px", height: "30px" }}
                  />
                </div>
                <h1 className="text-[#FFFFFFF0] text-base font-normal">
                  {faq.question}
                </h1>
                <MdKeyboardArrowRight
                  // size={24}
                  color="#FFFFFF"
                  style={{ width: "24px", height: "24px" }}
                />
              </Button>
            );
          })}
        </div>
      </section>
    </main>
  );
};

export default Home;
