import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { S } from "@/strings";

const qc = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <div className="h-full grid place-items-center text-muted-foreground">{S.appName}</div>
    </QueryClientProvider>
  );
}
