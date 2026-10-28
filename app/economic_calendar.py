"""Point-in-time economic calendar: what was knowable at a decision time, and nothing else.

The whole difficulty of an economic calendar is that every fact has at least three clocks -- when the event happens, when a
number is published, and when it reaches us -- and that later facts about the same event keep arriving for months. A feature
built from "the latest value" is built from information nobody had, and it will look excellent in a backtest for exactly that
reason.

This module stores ARRIVALS and answers questions AS OF a decision clock. Nothing is overwritten: a revision is a new arrival
that supersedes an earlier one for later views and leaves every earlier view exactly as it was. A surprise computed before a
release stays frozen at what was known then, however the consensus moves afterwards.

What it refuses rather than guesses, each because the alternative silently invents information:

* an ambiguous or naive timestamp -- a local wall clock during a DST fold names two instants, and picking one is a guess;
* mixed units or incomparable periods for the same series -- a "surprise" between a year-on-year percentage and a monthly
  index level is arithmetic, not economics;
* a missing consensus, or a historical residual scale of zero -- there is no standardized surprise to report, and reporting
  zero or infinity would put an invented number where a missing one belongs;
* an event whose historical availability is unknown -- the archive row is kept, and point-in-time use is refused;
* a computation that finishes after the decision deadline -- the result is stale, not backdated.

Nothing here reads the network, scrapes a provider or infers a release calendar from prices.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math

SCHEMA = "economic_calendar_arrival.v1"

#: an arrival is one of these. They are different facts about an event, not versions of one field.
ARRIVAL_KINDS = ("SCHEDULE", "CONSENSUS", "ACTUAL", "REVISION", "CANCELLATION", "SCHEDULE_UPDATE")

#: what must be present on every arrival. `historical_availability` is here, not in a helper, because a rule a caller can
#: forget to invoke is not a rule: an UNKNOWN row used to be ingested and produce a number anyway.
REQUIRED = ("schema", "event_key", "kind", "observed_at", "event_time", "historical_availability")

AVAILABILITY = ("KNOWN", "UNKNOWN")


class CalendarRefusal(ValueError):
    """A refusal that names its own reason. No calendar value is ever invented to avoid one."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def instant(value, field):
    """An unambiguous instant. A naive timestamp or a local wall clock inside a DST fold is refused, never resolved for you."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CalendarRefusal(f"NAIVE_TIMESTAMP: {field} has no offset, so it names no instant")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise CalendarRefusal(f"TIMESTAMP_REQUIRED: {field} is not a timestamp")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalendarRefusal(f"UNREADABLE_TIMESTAMP: {field}={value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalendarRefusal(
            f"AMBIGUOUS_LOCAL_TIME: {field}={value!r} carries no offset. During a daylight-saving fold one local wall clock "
            f"names two instants, and choosing one of them is a guess about when something was knowable")
    return parsed.astimezone(timezone.utc)


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CalendarRefusal(f"NON_FINITE_VALUE: {field}={value!r}")
    return float(value)


def validate_arrival(arrival):
    """One arrival, checked before it can enter the store. Returns a normalized copy."""
    if not isinstance(arrival, dict):
        raise CalendarRefusal("ARRIVAL_MUST_BE_A_MAPPING")
    missing = [f for f in REQUIRED if not arrival.get(f)]
    if missing == ["historical_availability"] or "historical_availability" in missing:
        raise CalendarRefusal(
            f"AVAILABILITY_MUST_BE_DECLARED: this arrival does not say whether its historical availability is KNOWN or "
            f"UNKNOWN{'' if missing == ['historical_availability'] else ', and also lacks ' + ', '.join(f for f in missing if f != 'historical_availability')}; "
            f"using a row as if it had been observable at its timestamp asserts something nobody verified")
    if missing:
        raise CalendarRefusal(f"ARRIVAL_FIELDS_MISSING: {', '.join(missing)}")
    if arrival["schema"] != SCHEMA:
        raise CalendarRefusal(f"UNKNOWN_ARRIVAL_SCHEMA: {arrival['schema']!r}")
    if arrival["kind"] not in ARRIVAL_KINDS:
        raise CalendarRefusal(f"UNKNOWN_ARRIVAL_KIND: {arrival['kind']!r} is not one of {list(ARRIVAL_KINDS)}")
    if arrival["historical_availability"] not in AVAILABILITY:
        raise CalendarRefusal(
            f"AVAILABILITY_MUST_BE_DECLARED: {arrival['historical_availability']!r} is not one of {list(AVAILABILITY)}; "
            f"using a row as if it had been observable at its timestamp asserts something nobody verified")
    out = dict(arrival)
    out["observed_at"] = instant(arrival["observed_at"], "observed_at")
    out["event_time"] = instant(arrival["event_time"], "event_time")
    if arrival.get("published_at"):
        out["published_at"] = instant(arrival["published_at"], "published_at")
        if out["published_at"] > out["observed_at"]:
            raise CalendarRefusal(
                f"RECEIVED_BEFORE_PUBLISHED: {arrival['event_key']} claims to have been observed at "
                f"{out['observed_at'].isoformat()}, before it was published at {out['published_at'].isoformat()}")
    for field in ("actual", "consensus", "previous"):
        if arrival.get(field) is not None:
            out[field] = _number(arrival[field], field)
    if out["kind"] in ("ACTUAL", "REVISION") and out.get("actual") is None:
        raise CalendarRefusal(f"{out['kind']}_WITHOUT_VALUE: {arrival['event_key']}")
    if out["kind"] == "CONSENSUS" and out.get("consensus") is None:
        raise CalendarRefusal(f"CONSENSUS_WITHOUT_VALUE: {arrival['event_key']}")
    for field in ("unit", "period"):
        if out["kind"] in ("ACTUAL", "REVISION", "CONSENSUS") and not arrival.get(field):
            raise CalendarRefusal(f"{field.upper()}_REQUIRED: a number without its {field} cannot be compared to another")
    out["sequence"] = arrival.get("sequence")
    if out["sequence"] is not None and (isinstance(out["sequence"], bool) or not isinstance(out["sequence"], int)):
        raise CalendarRefusal("SEQUENCE_MUST_BE_AN_INTEGER: it records observed order, not a timestamp")
    out["arrival_sha256"] = digest({k: (v.isoformat() if isinstance(v, datetime) else v)
                                    for k, v in out.items() if k != "arrival_sha256"})
    return out


class PointInTimeCalendar:
    """Arrivals in, as-of views out. An arrival is never modified and never removed."""

    def __init__(self):
        self._arrivals = []
        self._archive = []
        self._seen = set()

    # --- ingestion ----------------------------------------------------------------------------------------------
    def add(self, arrival):
        """Idempotent by content: the same arrival delivered twice, or out of order, is one arrival.

        A row whose historical availability is UNKNOWN is kept as ARCHIVE and never enters a point-in-time view. The refusal
        is here, on the only path in, rather than in a helper a caller may never call."""
        checked = validate_arrival(arrival)
        if checked["arrival_sha256"] in self._seen:
            return copy.deepcopy(checked), "DUPLICATE"
        self._seen.add(checked["arrival_sha256"])
        if checked["historical_availability"] == "UNKNOWN":
            self._archive.append(checked)
            return copy.deepcopy(checked), "ARCHIVED_NOT_POINT_IN_TIME"
        self._arrivals.append(checked)
        return copy.deepcopy(checked), "ADDED"

    def archive_rows(self):
        """Rows retained but refused for point-in-time use, detached so reading them cannot change them."""
        return copy.deepcopy(self._archive)

    def add_all(self, arrivals):
        return [self.add(a) for a in arrivals]

    def vintage_identity(self, as_of):
        """The identity of everything knowable at this instant. Order of delivery cannot change it."""
        cutoff = instant(as_of, "as_of")
        known = sorted(a["arrival_sha256"] for a in self._arrivals if a["observed_at"] <= cutoff)
        return digest({"as_of": cutoff.isoformat(), "arrivals": known})

    # --- views --------------------------------------------------------------------------------------------------
    def known_at(self, as_of, event_key=None):
        cutoff = instant(as_of, "as_of")
        rows = [a for a in self._arrivals if a["observed_at"] <= cutoff
                and (event_key is None or a["event_key"] == event_key)]
        # observed order breaks a tie between two arrivals stamped at the same instant; without one, neither is chosen.
        # The rows are COPIES: a public read must not be a handle on history, or a caller could change a value that has
        # already been hashed and the vintage identity would not move.
        return copy.deepcopy(sorted(rows, key=lambda a: (a["observed_at"],
                                                        a["sequence"] if a["sequence"] is not None else -1)))

    def view(self, event_key, as_of):
        """What was known about one event at one instant, with every value's own provenance."""
        rows = self.known_at(as_of, event_key)
        if not rows:
            return {"event_key": event_key, "as_of": instant(as_of, "as_of").isoformat(), "status": "UNKNOWN",
                    "reading": "no arrival about this event had been observed at that instant"}
        if any(r["kind"] == "CANCELLATION" for r in rows):
            return {"event_key": event_key, "as_of": instant(as_of, "as_of").isoformat(), "status": "CANCELLED",
                    "scheduled_for": rows[0]["event_time"].isoformat()}
        schedule = [r for r in rows if r["kind"] in ("SCHEDULE", "SCHEDULE_UPDATE")]
        consensus = [r for r in rows if r.get("consensus") is not None]
        actuals = [r for r in rows if r["kind"] in ("ACTUAL", "REVISION")]
        out = {"event_key": event_key,
               "as_of": instant(as_of, "as_of").isoformat(),
               "status": "SCHEDULED" if not actuals else "RELEASED",
               "event_time": (schedule[-1] if schedule else rows[0])["event_time"].isoformat(),
               "schedule_updates": len([r for r in schedule if r["kind"] == "SCHEDULE_UPDATE"]),
               "arrivals_known": len(rows),
               # the vintage is WHAT was known, not the order it arrived in: two deliveries of the same facts are one vintage
               "vintage_sha256": digest(sorted(r["arrival_sha256"] for r in rows))}
        if consensus:
            out["consensus"] = consensus[-1]["consensus"]
            out["consensus_observed_at"] = consensus[-1]["observed_at"].isoformat()
            out["unit"] = consensus[-1].get("unit")
            out["period"] = consensus[-1].get("period")
        if actuals:
            tied = [r for r in actuals if r["observed_at"] == actuals[-1]["observed_at"]]
            if len(tied) > 1 and any(r["sequence"] is None for r in tied):
                # CAL10: two values at the same clock with no observed order. Choosing one would be a coin flip presented
                # as a fact, so the conservative outcome is to report neither as current.
                out["status"] = "AMBIGUOUS_SEQUENCE"
                out["reading"] = ("two arrivals carry the same instant and no observed sequence; neither is reported as the "
                                  "current value and both are retained")
                out["tied_arrivals"] = [r["arrival_sha256"] for r in tied]
                return out
            current = actuals[-1]
            out["actual"] = current["actual"]
            out["actual_kind"] = current["kind"]
            out["actual_observed_at"] = current["observed_at"].isoformat()
            out["revisions_known"] = len([r for r in actuals if r["kind"] == "REVISION"])
            out["unit"] = current.get("unit")
            out["period"] = current.get("period")
            units = {r.get("unit") for r in actuals + consensus}
            periods = {r.get("period") for r in actuals + consensus}
            if len(units) > 1 or len(periods) > 1:
                raise CalendarRefusal(
                    f"INCOMPARABLE_SERIES: {event_key} mixes units {sorted(map(str, units))} and periods "
                    f"{sorted(map(str, periods))}; a difference between them would be arithmetic, not an economic surprise")
        return out

    # --- the surprise -------------------------------------------------------------------------------------------
    @staticmethod
    def _published(arrival):
        """When the world could have known it. A row without its own publication clock falls back to when we received it,
        which is the conservative direction: never earlier than it truly was."""
        return arrival.get("published_at") or arrival["observed_at"]

    def surprise(self, event_key, as_of, *, scale=None):
        """Two boundaries, reported side by side and never merged.

        `release_surprise` is what the market was surprised by: the actual against the last consensus PUBLISHED before the
        actual was published. `available_surprise` is what our own system could have computed: the same actual against the
        last consensus it had RECEIVED by the time the actual reached it.

        They differ exactly when a consensus is published after a release and arrives before we see the number -- and then
        the difference matters, because treating the later one as the expectation turns a real surprise into zero.
        """
        view = self.view(event_key, as_of)
        base = {"event_key": event_key, "as_of": view["as_of"], "status": view["status"],
                "vintage_sha256": view.get("vintage_sha256"),
                "boundaries": {
                    "release": "the consensus last PUBLISHED before the actual was published: what the market expected",
                    "available": "the consensus last RECEIVED here before the actual arrived here: what we could compute"}}
        if view["status"] != "RELEASED":
            return {**base, "release_surprise": None, "available_surprise": None,
                    "reason": f"NO_ACTUAL_YET: the event is {view['status'].lower()}"}
        rows = self.known_at(as_of, event_key)
        actuals = [r for r in rows if r["kind"] in ("ACTUAL", "REVISION")]
        consensus_rows = [r for r in rows if r.get("consensus") is not None]
        first_actual = actuals[0]
        release_pub = self._published(first_actual)
        before_publication = [r for r in consensus_rows if self._published(r) < release_pub]
        before_receipt = [r for r in consensus_rows if r["observed_at"] < first_actual["observed_at"]]
        out = {**base,
               "release_actual": first_actual["actual"],
               "release_actual_published_at": release_pub.isoformat(),
               "unit": first_actual.get("unit"), "period": first_actual.get("period"),
               "release_surprise": None, "available_surprise": None}
        current = actuals[-1]
        if current is not first_actual:
            # a revision is new information about the same event; it never rewrites what the release surprised anyone by
            out["revised_actual"] = current["actual"]
            out["revised_actual_kind"] = current["kind"]
            out["revised_observed_at"] = current["observed_at"].isoformat()
        for label, picked in (("release", before_publication), ("available", before_receipt)):
            if not picked:
                out[f"{label}_reason"] = ("NO_CONSENSUS_BEFORE_THE_BOUNDARY: there is no expectation to be surprised "
                                          "against, and reporting zero would put an invented number where a missing one "
                                          "belongs")
                continue
            chosen = picked[-1]
            if chosen.get("unit") != first_actual.get("unit") or chosen.get("period") != first_actual.get("period"):
                raise CalendarRefusal(
                    f"INCOMPARABLE_SERIES: {event_key} consensus is {chosen.get('unit')}/{chosen.get('period')} and the "
                    f"actual is {first_actual.get('unit')}/{first_actual.get('period')}")
            out[f"{label}_consensus"] = chosen["consensus"]
            out[f"{label}_consensus_published_at"] = self._published(chosen).isoformat()
            out[f"{label}_consensus_observed_at"] = chosen["observed_at"].isoformat()
            out[f"{label}_surprise"] = first_actual["actual"] - chosen["consensus"]
        if "revised_actual" in out and out.get("release_consensus") is not None:
            out["revision_surprise"] = current["actual"] - out["release_consensus"]
            out["revision_surprise_reading"] = ("the revised value against the SAME pre-release expectation; it is a "
                                                "different quantity from the release surprise, not an update of it")
        if scale is None:
            out["standardized"] = None
            out["standardized_reason"] = "NO_SCALE_SUPPLIED"
            return out
        scale = _number(scale, "scale")
        if scale <= 0:
            out["standardized"] = None
            out["standardized_reason"] = ("NON_POSITIVE_RESIDUAL_SCALE: dividing by it would be infinite or negative, and "
                                          "neither is a standardized surprise")
            return out
        out["scale"] = scale
        out["standardized"] = (out["release_surprise"] / scale) if out["release_surprise"] is not None else None
        out["standardized_boundary"] = "release"
        return out


def availability_checked(arrival, *, historical_availability):
    """Kept for callers that ask the question directly. The rule itself now lives on the ingestion path, where it cannot be
    skipped: `PointInTimeCalendar.add` archives an UNKNOWN row instead of admitting it."""
    """CAL09: point-in-time use requires knowing WHEN this became knowable. Unknown availability keeps the archive row and
    refuses the point-in-time use, rather than assuming the timestamp on the file is when somebody could have seen it."""
    if historical_availability not in ("KNOWN", "UNKNOWN"):
        raise CalendarRefusal("AVAILABILITY_MUST_BE_DECLARED: KNOWN or UNKNOWN")
    if historical_availability == "UNKNOWN":
        return {"point_in_time_usable": False,
                "archive_metadata_retained": True,
                "reason": ("UNKNOWN_HISTORICAL_AVAILABILITY: the row is kept as archive, and using it as if it had been "
                           "observable at its timestamp would assert something nobody verified")}
    return {"point_in_time_usable": True, "archive_metadata_retained": True}


def freshness(computed_at, decision_deadline):
    """CAL12: a result that lands after the deadline is stale. Backdating it would report a decision nobody could have made."""
    computed, deadline = instant(computed_at, "computed_at"), instant(decision_deadline, "decision_deadline")
    late = computed > deadline
    return {"computed_at": computed.isoformat(), "decision_deadline": deadline.isoformat(),
            "status": "STALE" if late else "IN_TIME",
            "usable_for_that_decision": not late,
            "late_by_seconds": max(0.0, (computed - deadline).total_seconds()),
            "reading": "a late result is reported as late; its timestamp is never moved back to the deadline"}
