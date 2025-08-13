import Heading from "@theme/Heading";
import clsx from "clsx";
import type { ReactNode } from "react";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "🚀 Simple AI Agent Framework",
    description: (
      <>
        Build your first agent with just a few lines of code — no boilerplate,
        no complex setup.
      </>
    ),
  },
  {
    title: "☁️ Cloud or Local",
    description: (
      <>
        Use a single API to work with both local and remote (API) models. That
        flexibility keeps you in full control of your stack.
      </>
    ),
  },
  {
    title: "💻 Cross-Platform & Multi-Language",
    description: (
      <>
        Supports Windows, Linux, and macOS — with developer-friendly APIs in
        Python and JavaScript.
      </>
    ),
  },
  {
    title: "🌎 WebAssembly Supports",
    description: (
      <>
        Run agents entirely in your web browsers — without any backend
        infrastructures
      </>
    ),
  },
];

function Feature({ title, description }: FeatureItem) {
  return (
    <div className={clsx("col col--6")}>
      <div className="padding-horiz--md">
        <Heading as="h2">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
