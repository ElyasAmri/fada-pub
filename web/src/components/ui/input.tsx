import * as React from "react";

import { cn } from "@/lib/utils";

export function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-primary disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}
