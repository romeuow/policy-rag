import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/Chat";
import AdminPage from "./pages/Admin";
import "./styles.css";

function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <strong>Assistente de Políticas</strong>
        <nav>
          <Link to="/">Chat</Link>
          <Link to="/admin">Admin</Link>
        </nav>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
