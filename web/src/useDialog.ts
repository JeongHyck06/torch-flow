// 창 하나의 키보드 규칙 (기획서 §2.2 접근성: 키보드만으로 완주).
//
// 세 가지가 창마다 같아야 한다: Esc로 닫히고, Tab이 창 밖으로 새지 않고,
// 닫으면 열기 전에 눌렀던 자리로 포커스가 돌아온다.

import { useEffect, useRef } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]),'
  + ' textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function useDialog<T extends HTMLElement>(open: boolean, close: () => void) {
  const box = useRef<T>(null);
  const opener = useRef<Element | null>(null);

  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement;
    // 창이 열리면 첫 조작 지점으로 간다 - 스크린 리더가 창 안에서 읽기 시작한다.
    const first = box.current?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        close();
        return;
      }
      if (event.key !== "Tab" || !box.current) return;
      const items = [...box.current.querySelectorAll<HTMLElement>(FOCUSABLE)]
        .filter((item) => item.offsetParent !== null);
      if (!items.length) return;
      const edge = event.shiftKey ? items[0] : items[items.length - 1];
      if (document.activeElement === edge || !box.current.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? items[items.length - 1] : items[0]).focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      (opener.current as HTMLElement | null)?.focus?.();
    };
  }, [open, close]);

  return box;
}
