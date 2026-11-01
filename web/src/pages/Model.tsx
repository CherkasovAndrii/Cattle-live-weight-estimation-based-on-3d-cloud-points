// What the model was trained on, how accurate it was in cross-validation,
// where it may be used, and the study figures when the server has them.
import { useState } from "react";
import { api } from "../api/client";
import { Card, ErrorBox, KeyValues, Loading, Stat } from "../components/ui";
import { useLoad } from "../lib/state";
import { fmt, when } from "../lib/text";

const HERD_UK: Record<string, string> = { cowdb2: "CowDatabase2, абердин-ангус", cowdb1: "CowDatabase, герефорд" };

export const FIGURES: [string, string][] = [
  ["01_pipeline_scheme", "Схема методу: від трьох хмар точок до маси з інтервалом"],
  ["03_segmentation_steps", "Виділення тварини: підлога, огорожа, тулуб"],
  ["05_cross_sections", "Поперечні перерізи тулуба, з яких складається об’єм"],
  ["09_within_herd_oof", "Оцінка проти ваг у перехресній перевірці"],
  ["10_prediction_intervals", "Інтервали прогнозу"],
  ["12_recalibration_curve", "Похибка на новому стаді залежно від кількості контрольних тварин"],
  ["16_sensor_tolerance", "Допустима похибка сенсора глибини"],
];

function Figure({ id, caption }: { id: string; caption: string }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <figure className="fig">
      <a href={`/figures/uk/${id}.png`} target="_blank" rel="noreferrer">
        <img src={`/figures/uk/${id}.png`} alt={caption} loading="lazy" onError={() => setOk(false)} />
      </a>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export function Model() {
  const { data: m, error, loading } = useLoad(() => api.model(), []);
  if (loading && !m) return <Loading />;
  if (!m) return <ErrorBox text={error} />;
  const cv = m.cross_validated;
  const half = parseFloat(String(m.interval).split("±")[1]);
  return (
    <div className="page">
      <div className="page-h"><h1>Модель</h1></div>
      <div className="stats">
        <Stat label="MAE, перехресна перевірка" value={fmt(cv.mae, 1)} unit="кг" />
        <Stat label="MAPE" value={fmt(cv.mape, 2)} unit="%" />
        <Stat label="90 % інтервал" value={`±${fmt(half, 1)}`} unit="кг" />
        <Stat label="Тварин у навчанні" value={fmt(m.n_training_animals)} />
      </div>
      <div className="grid2">
        <Card title="Картка моделі">
          <KeyValues rows={[
            ["Навчальні стада", m.training_herds.map((h) => HERD_UK[h] ?? h).join("; ")],
            ["Діапазон мас", `${fmt(m.weight_range_kg[0])}–${fmt(m.weight_range_kg[1])} кг`],
            ["Діапазон об’єму тулуба", `${fmt(m.trunk_volume_range_m3[0], 2)}–${fmt(m.trunk_volume_range_m3[1], 2)} м³`],
            ["Інтервал прогнозу", `${m.interval.split("%")[0]} % конформний, ±${fmt(half, 1)} кг`],
            ["RMSE", `${fmt(cv.rmse, 1)} кг`],
            ["Систематичний зсув", `${fmt(cv.bias, 1)} кг`],
            ["Найбільша похибка", `${fmt(cv.max_abs, 1)} кг`],
            ["Версія пакета", m.package_version],
            ["Навчено", when(m.created_utc)],
          ]} />
        </Card>
        <Card title="Коли оцінці не довіряють">
          <ul className="plain">
            <li>Об’єм тулуба поза діапазоном навчання: тварина менша або більша за ті, на яких навчено модель. Оцінку повернуто, але позначено.</li>
            <li>Стадо не відкалібровано: перенесення на нову ферму без контрольних зважувань дає систематичний зсув.</li>
            <li>Три хмари не сходяться в одну тварину або кадр зашумлений: система просить пересканувати.</li>
            <li>Похибка сенсора понад допустиму: шум глибини більше 0,75 % відстані або багато розсіяних пікселів без глибини.</li>
          </ul>
        </Card>
      </div>
      <Card title="Результати дослідження">
        <div className="figs">{FIGURES.map(([id, c]) => <Figure key={id} id={id} caption={c} />)}</div>
        <p className="small muted">Рисунки показує сервер з каталогу звітів. Якщо їх не видно, запустіть <code>cattle3d figures</code>.</p>
      </Card>
    </div>
  );
}
