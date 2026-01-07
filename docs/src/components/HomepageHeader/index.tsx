import Translate, { translate } from "@docusaurus/Translate";
import Heading from "@theme/Heading";
import clsx from "clsx";
import styles from "./style.module.css";

export default function HomepageHeader() {
  return (
    <header className={clsx("hero hero--dark", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          <Translate id="homepage.title">Build your AI Agents instantly</Translate>
        </Heading>
        <p className="hero__subtitle">
          <Translate id="homepage.tagline">in any environments, in any languages, without any complex setup.</Translate>
        </p>
        <a href="./docs/tutorial/getting-started">
          <button className="button button--primary button--outline button--lg">
            <Translate id="homepage.getStarted">Get Started</Translate>
          </button>
        </a>
      </div>
    </header>
  );
}
