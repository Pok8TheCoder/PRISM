import { Navigate, Routes, Route } from 'react-router-dom'
import { AppLayout } from './AppLayout'
import { DashboardPage } from '../features/dashboard/DashboardPage'
import { LabSessionPage } from '../features/lab-session/LabSessionPage'
import { AdversarialPage } from '../features/adversarial/AdversarialPage'
import { PlaceholderPage } from '../features/placeholder/PlaceholderPage'

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="session" element={<LabSessionPage />} />
        <Route path="lab-session" element={<Navigate to="/session" replace />} />
        <Route path="adversarial" element={<AdversarialPage />} />
        <Route
          path="explain"
          element={<PlaceholderPage title="Explain" description="SHAP waterfall on IPS blocks — phase 2" />}
        />
        <Route
          path="recordings"
          element={<PlaceholderPage title="Recordings" description="WebM + JSON manifest library — phase 2" />}
        />
      </Route>
    </Routes>
  )
}
