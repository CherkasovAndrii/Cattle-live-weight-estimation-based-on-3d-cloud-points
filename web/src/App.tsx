import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AppProvider } from "./lib/state";
import { About } from "./pages/About";
import { Animal, Animals } from "./pages/Animals";
import { Herd } from "./pages/Herd";
import { Model } from "./pages/Model";
import { Overview } from "./pages/Overview";
import { Weigh } from "./pages/Weigh";

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="weigh" element={<Weigh />} />
            <Route path="herd" element={<Herd />} />
            <Route path="animals" element={<Animals />} />
            <Route path="animals/:tag" element={<Animal />} />
            <Route path="model" element={<Model />} />
            <Route path="about" element={<About />} />
            <Route path="app/*" element={<Navigate to="/weigh" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}
