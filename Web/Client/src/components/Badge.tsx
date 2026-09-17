import React from "react";

interface BadgeProps {
  variant?: "success" | "info" | "warning" | "danger" | "neutral";
  children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({ variant = "neutral", children }) => {
  const styles = {
    success: "bg-[#00F59B]/15 text-[#00F59B] border-[#00F59B]/30",
    info: "bg-[#00B2FE]/15 text-[#00B2FE] border-[#00B2FE]/30",
    warning: "bg-[#FFB020]/15 text-[#FFB020] border-[#FFB020]/30",
    danger: "bg-[#FF4D4D]/15 text-[#FF4D4D] border-[#FF4D4D]/30",
    neutral: "bg-slate-800 text-slate-300 border-slate-700",
  }[variant];

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${styles}`}>
      {children}
    </span>
  );
};
