import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({ title, subtitle, action, children, className = "", ...props }) => {
  return (
    <div
      className={`bg-[#121A2B] border border-[#233354] rounded-xl p-4 shadow-lg flex flex-col ${className}`}
      {...props}
    >
      {(title || subtitle || action) && (
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#233354]/60">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-100 tracking-wide">{title}</h3>}
            {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="flex-1">{children}</div>
    </div>
  );
};
