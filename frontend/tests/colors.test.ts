import assert from "node:assert/strict";
import test from "node:test";

import {
  trackColor,
  ULTRALYTICS_PALETTE,
} from "../src/utils/colors.ts";


test("trackColor는 stride 7로 인접 ID를 팔레트 전체에 흩어 배치한다", () => {
  assert.equal(trackColor(1), "#3DDB86");
  assert.equal(trackColor(2), "#0018EC");
  assert.equal(trackColor(3), "#FF9D97");
  assert.equal(new Set(Array.from({ length: 20 }, (_, index) => trackColor(index))).size, 20);
});


test("trackColor는 동일·음수 ID에도 결정적인 팔레트 색을 반환한다", () => {
  assert.equal(trackColor(7), trackColor(7));
  assert.equal(trackColor(-1), ULTRALYTICS_PALETTE[13]);
});
