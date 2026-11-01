// How the system is put together: the station, the service, the library and
// this console, with a real merged cloud of one CowDatabase2 cow.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { CloudViewer, cloudFromBin, type Cloud } from "../components/CloudViewer";
import { Card } from "../components/ui";

const PARTS = [
  ["Станція зважування", "Три камери глибини Kinect v2 знімають тварину зліва, справа й згори в проході до доїльного залу або на виході з загону. Кожен кадр є хмарою точок 512×424."],
  ["Хмарний сервіс", "Сервер на FastAPI приймає три хмари, перевіряє їх, обчислює масу й зберігає результат у базі SQLite. Доступ захищено ключем."],
  ["Бібліотека cattle3d", "Одна функція аналізу спільна для сервісу, командного рядка й експериментів: підлога методом RANSAC, виділення тварини, система координат тіла, об’єм тулуба, ансамбль моделей, конформний інтервал."],
  ["Вебзастосунок", "Цей застосунок на React для комп’ютера й телефона: зважування, ферми, калібрування, контроль дрейфу, історія тварин і картка моделі."],
] as const;

const FLOW = [
  "Оператор обирає три хмари або демонстраційну тварину й вводить бирку.",
  "Застосунок надсилає файли на сервер і показує хід обробки.",
  "Сервер виділяє тварину, обчислює ознаки тулуба й масу з 90 % інтервалом.",
  "Правила рішення визначають статус: довірено, позначено або відмова.",
  "Результат зберігається в історії тварини, а зважування на вагах уточнюють калібрування й контроль дрейфу.",
];

export function About() {
  const [cloud, setCloud] = useState<Cloud | null>(null);
  useEffect(() => {
    fetch("/img/cow70_points.bin").then((r) => (r.ok ? r.arrayBuffer() : Promise.reject())).then((b) => setCloud(cloudFromBin(b, 2)), () => setCloud(null));
  }, []);
  return (
    <div className="page">
      <section className="hero">
        <div>
          <h1>Безконтактне визначення живої маси великої рогатої худоби</h1>
          <p className="lead">Система оцінює масу тварини за трьома хмарами точок з камер глибини, повертає 90 % інтервал і вирішує, чи можна довіряти оцінці. Нова ферма калібрується за 20–30 зважуваннями на вагах.</p>
          <div className="hero-actions">
            <Link to="/weigh" className="btn primary">Зважити тварину</Link>
            <Link to="/model" className="btn">Точність моделі</Link>
          </div>
        </div>
        {cloud && <CloudViewer cloud={cloud} height={300} controls={false} label="Об’єднана хмара точок корови з трьох камер" />}
      </section>
      <div className="stats">
        <div className="stat"><div className="stat-l">Герефорд, 101 тварина</div><div className="stat-v">16,1<small> кг MAE</small></div><div className="stat-n">MAPE 3,70 %</div></div>
        <div className="stat"><div className="stat-l">Абердин-ангус, 94 тварини</div><div className="stat-v">20,7<small> кг MAE</small></div><div className="stat-n">MAPE 3,34 %</div></div>
        <div className="stat"><div className="stat-l">Нова ферма</div><div className="stat-v">20–30<small> зважувань</small></div><div className="stat-n">для калібрування</div></div>
        <div className="stat"><div className="stat-l">Похибка сенсора</div><div className="stat-v">≤ 0,5–1<small> %</small></div><div className="stat-n">шум глибини без перекалібрування</div></div>
      </div>
      <div className="parts">
        {PARTS.map(([t, d], i) => (
          <Card key={t} title={<><span className="num-badge">{i + 1}</span>{t}</>}><p>{d}</p></Card>
        ))}
      </div>
      <Card title="Як проходить одне зважування">
        <ol className="flow">{FLOW.map((f) => <li key={f}>{f}</li>)}</ol>
      </Card>
    </div>
  );
}
