import React from "react";

function SmallIcon(props: { src: string }) {
  return (
    <img
      src={props.src}
      width={20}
      height={20}
      style={{ verticalAlign: "middle" }}
      alt="small-icon"
    />
  );
}

export default SmallIcon;
