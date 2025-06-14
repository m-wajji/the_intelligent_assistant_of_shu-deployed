import React, { useEffect, useRef, useState } from "react";
import { Play, Pause } from "lucide-react";

interface Props {
  src: string;
  onEnded: () => void;
  autoplay?: boolean;
}

const AudioMessage: React.FC<Props> = ({ src, onEnded, autoplay = false }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const onEndedRef = useRef(onEnded);
  const autoplayRef = useRef(autoplay);

  useEffect(() => {
    onEndedRef.current = onEnded;
  }, [onEnded]);

  useEffect(() => {
    autoplayRef.current = autoplay;
  }, [autoplay]);

  // Initialize audio element only once when src changes
  useEffect(() => {
    const audio = new Audio(src);
    audioRef.current = audio;
    audio.playbackRate = 1.2;

    const handleTimeUpdate = () => {
      if (audio.duration > 0) {
        setProgress((audio.currentTime / audio.duration) * 100);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setProgress(0);
      onEnded();
    };


    const handleLoadedData = () => {
      if (autoplay) {
        audio
          .play()
          .then(() => setIsPlaying(true))
          .catch((err) => {
            console.error("Error playing audio:", err);
            setIsPlaying(false);
          });
      }
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("loadeddata", handleLoadedData);

    return () => {
      // Cleanup: remove listeners and pause audio
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("loadeddata", handleLoadedData);

      if (!audio.paused) {
        audio.pause();
      }
      setIsPlaying(false);
    };
  }, [src]); // Only depend on src, not autoplay and onEnded

  // Handle autoplay changes (like when recording starts)
  useEffect(() => {
    if (!autoplay && audioRef.current && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else if (
      autoplay &&
      audioRef.current &&
      !isPlaying &&
      audioRef.current.readyState >= 2
    ) {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => {
          console.error("Error playing audio:", err);
          setIsPlaying(false);
        });
    }
  }, [autoplay]);

  const togglePlayback = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      // Reset to beginning if audio has ended
      if (audioRef.current.ended) {
        audioRef.current.currentTime = 0;
        setProgress(0);
      }

      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((err) => {
          console.error("Error playing audio:", err);
          setIsPlaying(false);
        });
    }
  };

  // const formatTime = (seconds: number) => {
  //   if (isNaN(seconds) || !isFinite(seconds) || seconds <= 0) return "0:00";
  //   const mins = Math.floor(seconds / 60);
  //   const secs = Math.floor(seconds % 60);
  //   return `${mins}:${secs.toString().padStart(2, "0")}`;
  // };

  return (
    <div className="flex mb-4 items-center space-x-3">
      <button
        onClick={togglePlayback}
        className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 focus:outline-none transition-colors"
        disabled={!audioRef.current}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
      </button>

      <div className="flex-1 max-w-xs">
        <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className="bg-blue-500 h-full rounded-full transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default AudioMessage;
