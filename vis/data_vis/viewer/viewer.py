"""OpenCV 交互窗口：帧滑条 + 键盘播放/单步/图层切换/截图。

模块: vis/data_vis/viewer/viewer.py
依赖: cv2, time, vis.data_vis.draw, vis.data_vis.viewer.checks.viewer_checks
读取配置: cfg.data_vis.display(window_name/play_fps/scale)、cfg.data_vis.bbox.draw_static（静态框初始开关）
对外接口:
    - Viewer(reader, vcfg).run() -> None     # 阻塞式事件循环，窗口关闭/按 q 退出
说明: 仅负责窗口与交互；每帧的实际合成委托 vis.data_vis.draw.render_frame。显示开关按 reader.available
      动态建立——切换不存在的模态无效果且不在 HUD 出现。仅当帧号或图层开关变化时重渲染（脏标记），
      播放节奏按 display.play_fps 计时推进。截图落盘到当前工作目录。
"""

import time

import cv2

from vis.data_vis import draw
from vis.data_vis.viewer.checks.viewer_checks import check_viewer

_QUIT_KEYS = {ord("q"), 27}            # q / Esc
_LEFT_KEYS = {ord("a"), ord(","), 2424832}   # 上一帧（含 Windows 左方向键扫描码）
_RIGHT_KEYS = {ord("."), 2555904}            # 下一帧（含右方向键；'d' 留给深度开关）
# 按键 -> state 开关名（模态开关；不存在的模态切换无效果，由 draw 按 available 过滤）
_TOGGLE_KEYS = {ord("b"): "show_bbox", ord("s"): "show_static", ord("r"): "show_rgb",
                ord("d"): "show_depth", ord("m"): "show_semantic", ord("f"): "show_flow",
                ord("v"): "show_bev"}


class Viewer:
    def __init__(self, reader, vcfg):
        check_viewer(reader)
        self._reader = reader
        self._vcfg = vcfg
        self._win = vcfg.display.window_name
        self._playing = False
        self._dirty = True
        self._sync = False  # 程序化设置滑条时屏蔽其回调，避免递归
        # 各模态默认开启；available 透传给 draw，决定哪些层真正渲染与在 HUD 列出
        self._state = {"show_bbox": True, "show_static": vcfg.bbox.draw_static,
                       "show_rgb": True, "show_depth": True, "show_semantic": True,
                       "show_flow": True, "show_bev": True,
                       "available": reader.available, "playing": False,
                       "idx": 0, "num_frames": reader.num_frames}

    def run(self):
        """进入事件循环；窗口关闭或按 q/Esc 返回。"""
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("frame", self._win, 0, max(1, self._reader.num_frames - 1),
                           self._on_track)
        interval = 1.0 / self._vcfg.display.play_fps
        last = time.time()
        while True:
            if self._dirty:
                self._show()
                self._dirty = False
            key = cv2.waitKeyEx(15)
            if key != -1 and not self._handle_key(key):
                break
            if self._playing and time.time() - last >= interval:
                last = time.time()
                self._advance(1)
            if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    def _show(self):
        self._state["playing"] = self._playing  # 让 HUD 反映播放/暂停态
        frame = self._reader.frame(self._state["idx"])
        canvas = draw.render_frame(frame, self._reader.meta, self._vcfg, self._state)
        scale = self._vcfg.display.scale
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(self._win, canvas)

    def _on_track(self, pos):
        if self._sync:
            return
        self._state["idx"] = pos
        self._playing = False
        self._dirty = True

    def _advance(self, step):
        self._state["idx"] = (self._state["idx"] + step) % self._reader.num_frames
        self._sync = True
        cv2.setTrackbarPos("frame", self._win, self._state["idx"])
        self._sync = False
        self._dirty = True

    def _handle_key(self, key):
        """处理一次按键；返回 False 表示退出。"""
        if key in _QUIT_KEYS:
            return False
        if key == 32:                       # 空格：播放/暂停
            self._playing = not self._playing
            self._dirty = True              # 立即刷新 HUD 的播放态
        elif key in _LEFT_KEYS:
            self._playing = False
            self._advance(-1)
        elif key in _RIGHT_KEYS:
            self._playing = False
            self._advance(1)
        elif key in _TOGGLE_KEYS:
            self._toggle(_TOGGLE_KEYS[key])
        elif key == ord("w"):
            self._screenshot()
        return True

    def _toggle(self, name):
        self._state[name] = not self._state[name]
        self._dirty = True

    def _screenshot(self):
        frame = self._reader.frame(self._state["idx"])
        canvas = draw.render_frame(frame, self._reader.meta, self._vcfg, self._state)
        out = "{}_f{:04d}.png".format(self._reader.meta["scene_id"], self._state["idx"])
        cv2.imwrite(out, canvas)
        print("[vis] 截图已保存:", out)
