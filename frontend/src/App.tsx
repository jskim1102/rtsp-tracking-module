import CamerasPage from "./pages/CamerasPage";
import MockTrackingPreview from "./dev/MockTrackingPreview";

export default function App() {
  if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("mock")) {
    return <MockTrackingPreview />;
  }
  return <CamerasPage />;
}
