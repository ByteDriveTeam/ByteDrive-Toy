const header = document.querySelector("[data-header]");
const menuButton = document.querySelector("[data-menu-button]");
const menu = document.querySelector("[data-menu]");

const setHeader = () => header.classList.toggle("scrolled", window.scrollY > 20);
setHeader();
window.addEventListener("scroll", setHeader, { passive: true });

menuButton.addEventListener("click", () => {
  const open = menu.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.setAttribute("aria-label", open ? "关闭导航" : "打开导航");
});
menu.addEventListener("click", (event) => {
  if (event.target.closest("a")) {
    menu.classList.remove("open");
    menuButton.setAttribute("aria-expanded", "false");
  }
});

const steps = [
  {
    eyebrow: "CARLA SYNTHETIC DATA",
    title: "异构双进程采集",
    body: "Python 3.7 worker 驱动 CARLA，Python 3.12 collector 负责编排、编码与落盘；共享内存完成帧数据交接。",
    list: ["RGB / Depth / Semantic / Optical Flow / LiDAR", "传感器 2Hz · 运动学 10Hz 异频采样", "H.265 视频 + LMDB 非 RGB 数据"],
  },
  {
    eyebrow: "FROZEN DINOv3 + MULTI-TASK",
    title: "语义与深度感知",
    body: "冻结 DINOv3 ViT-S+/16，融合浅、中、深层完整 Token，同时建立语义类别与米制深度表征。",
    list: ["第 3 / 6 / 12 层完整 Token", "29 类语义分割 + Symlog 深度回归", "感知表征复用于驾驶网络"],
  },
  {
    eyebrow: "GEOMETRY-AWARE BEV",
    title: "多模态联合规划",
    body: "三路相机 frustum、LiDAR 体素与历史 BEV 在前向鸟瞰空间中汇合，多任务分支共同约束可驾驶表征。",
    list: ["风险 / 可行驶 / 轨迹分布三场", "道路线、停止线与交通灯状态", "8 模态 × 20 个 10Hz 航点"],
  },
  {
    eyebrow: "SYNCHRONOUS CLOSED LOOP",
    title: "推理、重排与控制",
    body: "模型进入 CARLA 同步仿真；候选轨迹经规则安全重排后，由纯追踪与速度 PID 转换为车辆控制量。",
    list: ["双帧在线推理与历史状态维护", "候选轨迹安全重排", "Episode 终态、进度、控制与录像"],
  },
];

const tabs = [...document.querySelectorAll("[data-step]")];
const panelNumber = document.querySelector("[data-panel-number]");
const panelEyebrow = document.querySelector("[data-panel-eyebrow]");
const panelTitle = document.querySelector("[data-panel-title]");
const panelBody = document.querySelector("[data-panel-body]");
const panelList = document.querySelector("[data-panel-list]");

tabs.forEach((tab) => tab.addEventListener("click", () => {
  const index = Number(tab.dataset.step);
  const step = steps[index];
  tabs.forEach((item) => {
    const active = item === tab;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  panelNumber.textContent = String(index + 1).padStart(2, "0");
  panelEyebrow.textContent = step.eyebrow;
  panelTitle.textContent = step.title;
  panelBody.textContent = step.body;
  panelList.replaceChildren(...step.list.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
}));

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("in-view");
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));

const dialog = document.querySelector("[data-lightbox-dialog]");
const dialogImage = dialog.querySelector("img");
document.querySelectorAll("[data-lightbox]").forEach((button) => button.addEventListener("click", () => {
  dialogImage.src = button.dataset.lightbox;
  dialog.showModal();
}));
document.querySelector("[data-lightbox-close]").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

const copyButton = document.querySelector("[data-copy]");
copyButton.addEventListener("click", async () => {
  const command = ".\\.venv\\Scripts\\python.exe clone_loop\\run.py --env release20260803 --max-episodes 1";
  try {
    await navigator.clipboard.writeText(command);
    copyButton.textContent = "已复制";
    setTimeout(() => { copyButton.textContent = "复制"; }, 1600);
  } catch {
    copyButton.textContent = "复制失败";
  }
});
