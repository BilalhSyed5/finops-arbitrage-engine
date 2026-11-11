import os
import subprocess
import unittest
from datetime import datetime, timedelta

from finops.actions import render_ec2_script
from finops.infra_audit import analyze_infra, load_inventory, load_utilization
from finops.models import Confidence
from tests.helpers import BOOK, SAMPLE, write_tmp

START = datetime(2026, 10, 26)  # a Monday


def series(fn, days=14):
    return [(START + timedelta(hours=h), *fn(START + timedelta(hours=h))) for h in range(days * 24)]


def inv(iid, itype, **kw):
    row = {"instance_id": iid, "instance_type": itype, "name": "", "team": "",
           "avg_cpu": None, "avg_gpu": None, "running_hours": None, "hourly_rate": None}
    row.update(kw)
    return row


class TestTimeSeries(unittest.TestCase):
    def test_business_hours_gpu_gets_measured_schedule(self):
        def load(ts):
            busy = ts.weekday() < 5 and 9 <= ts.hour < 18
            return (40.0, 70.0) if busy else (1.0, 0.5)
        r = analyze_infra([inv("i-1", "g5.12xlarge")], BOOK, {"i-1": series(load)})
        (f,) = r.findings
        self.assertEqual(f.method, "timeseries")
        self.assertEqual(f.confidence, Confidence.MEASURED)
        self.assertTrue(f.schedule.weekend_off)
        self.assertEqual((f.schedule.weekday_stop_hour, f.schedule.weekday_start_hour), (18, 9))
        self.assertAlmostEqual(f.schedule.hours_saved_per_week, 168 - 45)
        self.assertEqual(f.schedule.busy_hours_in_window_per_week, 0)
        stops, starts = f.schedule.transitions()
        self.assertEqual(sorted(stops), [(d, 18) for d in range(5)])
        self.assertEqual(sorted(starts), [(d, 9) for d in range(5)])

    def test_zombie(self):
        r = analyze_infra([inv("i-z", "p4d.24xlarge")], BOOK, {"i-z": series(lambda ts: (1.0, 0.0))})
        (f,) = r.findings
        self.assertTrue(f.zombie)
        self.assertAlmostEqual(f.monthly_savings, f.monthly_cost)
        self.assertEqual(f.severity, "critical")

    def test_oversized_gpu_is_right_sized_and_estimated(self):
        r = analyze_infra([inv("i-o", "g6.12xlarge")], BOOK, {"i-o": series(lambda ts: (9.0, 14.0))})
        (f,) = r.findings
        self.assertIsNone(f.schedule)
        self.assertEqual(BOOK.instance(f.rightsize_to).gpus, 1)
        self.assertEqual(f.overall_confidence, Confidence.ESTIMATED)

    def test_busy_instance_is_healthy(self):
        r = analyze_infra([inv("i-h", "m5.2xlarge")], BOOK, {"i-h": series(lambda ts: (55.0, None))})
        self.assertEqual(r.healthy, ["i-h"])
        self.assertEqual(r.monthly_savings, 0)


class TestAverageMode(unittest.TestCase):
    def test_empty_cells_do_not_crash(self):
        # v1 bug: an empty hourly_rate cell aborted the whole run.
        path = write_tmp("instance_id,instance_type,avg_cpu_utilization_pct,avg_gpu_utilization_pct,"
                         "running_hours_per_day,hourly_rate\n"
                         "i-1,g5.xlarge,5,3,24,\n"
                         "i-2,m5.xlarge,,,,\n"
                         ",m5.xlarge,5,,24,\n", ".csv")
        rows, rep = load_inventory(path)
        self.assertEqual(rep.skipped, 1)
        r = analyze_infra(rows, BOOK)
        self.assertEqual(r.findings[0].instance_id, "i-1")
        self.assertEqual(r.findings[0].confidence, Confidence.ESTIMATED)
        self.assertTrue(any("i-2" in w for w in r.warnings))

    def test_inventory_rate_overrides_list_price(self):
        r = analyze_infra([inv("i-1", "g5.xlarge", avg_cpu=3.0, avg_gpu=1.0, running_hours=24.0,
                               hourly_rate=0.5)], BOOK)
        self.assertAlmostEqual(r.findings[0].hourly_rate, 0.5)

    def test_unknown_instance_type_reported(self):
        r = analyze_infra([inv("i-x", "z9.mega", avg_cpu=3.0)], BOOK)
        self.assertEqual(r.unpriced, ["i-x (z9.mega)"])


class TestDemoInfraAndScript(unittest.TestCase):
    def test_demo_files_and_generated_script_is_valid_bash(self):
        rows, _ = load_inventory(os.path.join(SAMPLE, "inventory.csv"))
        ser, types, _ = load_utilization(os.path.join(SAMPLE, "utilization.csv"))
        r = analyze_infra(rows, BOOK, ser, types)
        self.assertGreater(r.monthly_savings, 0)
        script = render_ec2_script(r)
        path = write_tmp(script, ".sh")
        self.assertEqual(subprocess.run(["bash", "-n", path]).returncode, 0)
        env = dict(os.environ, ROLE_ARN="arn:aws:iam::1:role/s", DRY_RUN="1")
        out = subprocess.run(["bash", path], capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("aws scheduler create-schedule", out.stdout)


if __name__ == "__main__":
    unittest.main()
