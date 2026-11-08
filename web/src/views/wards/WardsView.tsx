import { useData, useTwin } from '../../state/store'
import { Copilot } from './Copilot'
import { ResponseOrder } from './ResponseOrder'
import { WardList } from './WardList'
import { WardProfile } from './WardProfile'

const COL = 'min-[1100px]:h-full min-[1100px]:overflow-y-auto min-[1100px]:overscroll-contain'

export function WardsView() {
  const data = useData()
  const wardCode = useTwin((s) => s.wardCode)
  const ward = data.wards.wards.find((w) => w.code === wardCode) ?? data.wards.wards[0]

  if (!ward) {
    return (
      <div className="flex h-full items-center justify-center pt-20 text-sm text-mute">Ward data did not load. Reload the page to try again.</div>
    )
  }

  return (
    <div className="absolute inset-0 overflow-y-auto pt-20 min-[1100px]:overflow-hidden">
      <div className="mx-auto grid max-w-[1760px] grid-cols-1 gap-4 min-[1100px]:h-full min-[1100px]:grid-cols-[340px_minmax(0,1fr)_400px] min-[1440px]:grid-cols-[340px_minmax(0,1fr)_440px] min-[1100px]:gap-2 min-[1100px]:px-3">
        <aside aria-label="Ward ranking" className={`${COL} max-h-[60vh] overflow-y-auto border-b border-line min-[1100px]:max-h-none min-[1100px]:border-b-0 min-[1100px]:border-r`}>
          <WardList />
        </aside>
        <section aria-label={`Ward ${ward.code} profile`} className={COL}>
          <WardProfile ward={ward} />
        </section>
        <aside aria-label="Response order and copilot" className={`${COL} space-y-4 px-4 pb-6 pt-1 min-[1100px]:px-2 min-[1100px]:pr-1`}>
          <ResponseOrder ward={ward} />
          <Copilot ward={ward} />
        </aside>
      </div>
    </div>
  )
}
