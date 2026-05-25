import * as React from "react";

import { cn } from "@/lib/utils";

export function Textarea({
  className,
  ...props
}: React.ComponentProps<"textarea">) {
  return (
    <textarea
      className={cn(
        "flex min-h-24 w-full resize-none rounded-2xl border border-border bg-card px-4 py-3 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-primary disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}
