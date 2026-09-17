import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  size = "md",
  className = "",
  children,
  ...props
}) => {
  const sizeClasses = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-5 py-2.5 text-base",
  }[size];

  const variantClasses = {
    primary: "bg-[#00F59B] text-slate-950 font-semibold hover:bg-[#00F59B]/90 shadow-md shadow-[#00F59B]/20",
    secondary: "bg-[#182238] text-slate-100 hover:bg-[#233354] border border-[#233354]",
    outline: "border border-[#00F59B] text-[#00F59B] hover:bg-[#00F59B]/10",
    ghost: "text-slate-300 hover:bg-[#182238] hover:text-white",
  }[variant];

  return (
    <button
      className={`inline-flex items-center justify-center rounded-lg font-medium transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed ${sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
