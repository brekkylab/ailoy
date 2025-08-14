import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import CodePreview from "@site/src/components/CodePreview";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import HomepageHeader from "@site/src/components/HomepageHeader";
import Layout from "@theme/Layout";
import type { ReactNode } from "react";

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout title="Ailoy" description={siteConfig.title}>
      <HomepageHeader />
      <main
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-around",
          gap: "2rem",
          padding: "2rem 0px",
        }}
      >
        <HomepageFeatures />
        <CodePreview />
      </main>
    </Layout>
  );
}
