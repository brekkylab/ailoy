import Translate from "@docusaurus/Translate";
import Heading from "@theme/Heading";
import clsx from "clsx";
import type { ReactNode } from "react";
import styles from "./styles.module.css";

type FeatureItem = {
  titleId: string;
  titleDefault: string;
  descriptionId: string;
  descriptionDefault: string;
};

const FeatureList: FeatureItem[] = [
  {
    titleId: "homepage.feature1.title",
    titleDefault: "🚀 Simple AI Agent Framework",
    descriptionId: "homepage.feature1.description",
    descriptionDefault: "Build your first agent with just a few lines of code — no boilerplate, no complex setup.",
  },
  {
    titleId: "homepage.feature2.title",
    titleDefault: "☁️ Cloud or Local",
    descriptionId: "homepage.feature2.description",
    descriptionDefault: "Use a single API to work with both local and remote (API) models. That flexibility keeps you in full control of your stack.",
  },
  {
    titleId: "homepage.feature3.title",
    titleDefault: "💻 Cross-Platform & Multi-Language",
    descriptionId: "homepage.feature3.description",
    descriptionDefault: "Supports Windows, Linux, and macOS — with developer-friendly APIs in Python and JavaScript.",
  },
  {
    titleId: "homepage.feature4.title",
    titleDefault: "🌎 WebAssembly Supports",
    descriptionId: "homepage.feature4.description",
    descriptionDefault: "Run agents entirely in your web browsers — without any backend infrastructures",
  },
];

function Feature({ titleId, titleDefault, descriptionId, descriptionDefault }: FeatureItem) {
  return (
    <div className={clsx("col col--6")}>
      <div className="padding-horiz--md">
        <Heading as="h2">
          <Translate id={titleId}>{titleDefault}</Translate>
        </Heading>
        <p>
          <Translate id={descriptionId}>{descriptionDefault}</Translate>
        </p>
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
