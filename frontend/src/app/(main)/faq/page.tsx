"use client";
import { useState } from "react";
import { FaPlus, FaMinus } from "react-icons/fa6";

const FAQs = () => {
  const [activeFaq, setActiveFaq] = useState<number | null>(null);

  const toggleFaq = (id: number) => {
    setActiveFaq(activeFaq === id ? null : id);
  };
  const faqs = [
    {
      id: 1,
      question: "Can I apply to multiple programs/fields?",
      answer:
        "Yes, you may apply to multiple disciplines (up to 3). If all seats in the program that is your first priority are already filled, you may be offered admission in the other fields you opted for, subject to availability of seats.",
    },
    {
      id: 2,
      question: "Can I apply for admission online?",
      answer:
        "Yes, you may apply online by visiting our admissions page at www.shu.edu.pk/admissions. The highly user-friendly interface will take you through the application process step-by-step.",
    },
    {
      id: 3,
      question:
        "Is there a word limit for the Essay section of the Aptitude Test?",
      answer:
        "There is no maximum word count for the essay on the admission test. The students are provided with 4 pages (A4 size) to write the essay. Minimum word count is 100 words but students are encouraged to write more words in order to present a cogent logical essay.",
    },
    {
      id: 4,
      question: "What is the eligibility criteria for admissions?",
      answer:
        "For the undergraduate programs in Biomedical Engineering and Pharmacy, the minimum requirement is 60%; for BS Computer Science, it is 50% marks in HSC, and for BS Biosciences, BBA, and BS Accounting and Finance, it is 45%. You may find our Eligibility Criteria in detail at Salim Habib University Admission Eligibility Criteria (shu.edu.pk).",
    },
    {
      id: 5,
      question: "How can I prepare for admission test?",
      answer:
        "The admission test primarily tests your Verbal and Analytical skills, as well as general knowledge and science basics. A model paper with sample questions is available at www.shu.edu.pk/admissions/sample-test-paper. It provides clear guidelines on subject areas, number of questions, and duration of the test.",
    },
    {
      id: 6,
      question: "Can students awaiting results apply?",
      answer:
        "Yes, students who are waiting for their results may apply. However, they will be given provisional admission, subject to submitting their results by the prescribed date.",
    },
    {
      id: 7,
      question: "My percentage is below 50. Can I apply?",
      answer:
        "Any candidate whose percentage is below 50% is not eligible for admission in the Doctor of Pharmacy, BE Biomedical Engineering, and BS Computer Science Programs. You may apply for our BS Biosciences, BBA, and BS Accounting and Finance, given your percentage is above 45% and you meet other eligibility criteria.",
    },
    {
      id: 8,
      question: "Where is SHU located?",
      answer:
        "The address of Salim Habib University (Formerly Barrett Hodgson University) is as follows:\
         Salim Habib University (Formerly Barrett Hodgson University),\
         NC – 24, Deh Dih Korangi Creek, Karachi 74900,\
         Tel: +92-21-35122931-35,\
         Direction Map: https://goo.gl/maps/F4W44qDDV8H2",
    },
    {
      id: 9,
      question: "Can I book a Campus Visit?",
      answer:
        "Yes, the university arranges campus trips on a periodical basis. In addition to scheduled campus visits, specially arranged trips are possible if notified well in advance; exact dates and times are communicated to the students after they have filled the online form available at www.shu.edu.pk/campus-tour. For further details regarding campus tours, you may contact the Marketing and Communications Department at Tel: 021-35122931 Ext 221/226",
    },
    {
      id: 10,
      question:
        "What is the course waiver/transfer policy for transfer candidates?",
      answer:
        "The University may consider waiving off the SHU Aptitude Test on a case-to-case basis for such candidates; however the concerned candidates will be required to appear for interview and also provide valid transcripts of the courses undertaken/credits achieved at their previous institution. The decision of whether or not to accept any transfer credits rests with the Transfer Committee.As a policy guideline, a minimum of 50% of total degree credits must be completed at SHU, and the courses transferred shall be recorded in the student’s Final Transcript as being transfer credits.",
    },
    {
      id: 11,
      question: "Is there any scholarship available?",
      answer:
        "Yes, several merit-based scholarships are available, such as Dr. Salim Habib Scholarship (100%), Dr. Iram Afaq Scholarship (50%). We also offer financial aid. Visit our scholarship page at www.shu.edu.pk/fee-structure for more details.",
    },
    {
      id: 12,
      question: "Do you provide transport service?",
      answer:
        "Yes, SHU provides third party transport services to all parts of the city.",
    },
  ];
  return (
    <main>
      <header className="p-16 flex items-center justify-center">
        <h1 className="text-2xl font-bold text-[#ada7a7cc]">
          Frequently Asked Questions
        </h1>
      </header>

      <section className="flex flex-col items-center min-h-screen pb-5">
        <div className="flex flex-col w-1/2 gap-4">
          {faqs.map((faq) => (
            <div key={faq.id} className="faq-item">
              <div
                className="faq-question flex justify-between items-center bg-[#FF000070] h-16 px-4 rounded-xl "
                onClick={() => toggleFaq(faq.id)}
              >
                <h2 className="text-lg font-semibold">{faq.question}</h2>
                <span>
                  {activeFaq === faq.id ? (
                    <FaMinus size={24} />
                  ) : (
                    <FaPlus size={24} />
                  )}
                </span>
              </div>
              {activeFaq === faq.id && (
                <div className="faq-answer py-2 px-6 bg-slate-100 rounded-xl">
                  <p className="text-black">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
};

export default FAQs;
