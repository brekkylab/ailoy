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
    title: "🚀 Simple by Design",
    description: (
      <>
        Run your first LLM with just a few lines of code — no boilerplate, no
        complex setup.
      </>
    ),
  },
  {
    title: "☁️ Cloud or On-Device",
    description: (
      <>
        Use the same API to run large models in the cloud or optimized ones
        directly on-device. That flexibility keeps you in full control of your
        stack.
      </>
    ),
  },
  {
    title: "💻 Cross-Platform & Multi-Language",
    description: (
      <>
        Supports Windows, Linux, and macOS — with clean APIs for Python and
        JavaScript.
      </>
    ),
  },
];

function Feature({ title, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
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
