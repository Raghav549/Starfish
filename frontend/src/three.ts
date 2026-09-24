import * as THREE from "three";

export type ScanScene = {
  renderer: THREE.WebGLRenderer;
  frame: (progress: number, state: "idle" | "scanning" | "done") => void;
  dispose: () => void;
};

export function mountScanScene(canvas: HTMLCanvasElement): ScanScene {
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(canvas.clientWidth || 320, canvas.clientHeight || 180, false);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1.6, 0.1, 100);
  camera.position.set(0, 0, 7);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(2.0, 0.045, 16, 96),
    new THREE.MeshBasicMaterial({color: 0x4c9fff})
  );
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 24, 24),
    new THREE.MeshBasicMaterial({color: 0x111827, transparent: true, opacity: 0.85})
  );
  scene.add(ring, core);

  const scan = new THREE.Mesh(
    new THREE.PlaneGeometry(4.4, 0.035),
    new THREE.MeshBasicMaterial({color: 0x0ea5e9, transparent: true, opacity: 0.75})
  );
  scan.position.y = -2.1;
  scene.add(scan);

  const frame = (progress: number, state: "idle" | "scanning" | "done") => {
    ring.rotation.z = progress * Math.PI * 2;
    ring.scale.setScalar(0.94 + Math.sin(progress * Math.PI * 8) * 0.03);
    scan.position.y = -2.1 + progress * 4.2;
    if (state === "done") {
      scan.position.y = 2.1;
      ring.scale.setScalar(1.03);
    }
    renderer.render(scene, camera);
  };

  const resize = () => {
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 180;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    frame(0, "idle");
  };
  window.addEventListener("resize", resize);
  resize();

  return {
    renderer,
    frame,
    dispose: () => {
      window.removeEventListener("resize", resize);
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
      core.geometry.dispose();
      (core.material as THREE.Material).dispose();
      scan.geometry.dispose();
      (scan.material as THREE.Material).dispose();
      renderer.dispose();
    }
  };
}
