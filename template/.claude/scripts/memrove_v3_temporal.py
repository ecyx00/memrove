#!/usr/bin/env python3
"""Router rule R2: parse the time expression in a query into a (since, until) date range. Stdlib only, no model.

parse(query, ref_date) -> (since, until, (start, end)) | None
  since/until: datetime.date, inclusive. (start, end): char offsets of the matched text in `query`.
Precision beats coverage: a vague or unknown expression ("recently", "geçenlerde", "a few days ago") returns None;
an ambiguous one ("last Tuesday" said on a Wednesday) returns a window that holds every reading.
Point-in-time "N units ago" is centred on the date with a tolerance of half the unit (day 0, week 3, month 15, year 182 days).
Several expressions in one query give the union of their ranges (a superset of any one constraint).
"""
import calendar
import re
from datetime import date, timedelta

_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i", "’": "'", "‘": "'", "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g",
                       "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u"})
_TR_NUM = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
           "yirmi": 20, "otuz": 30, "kirk": 40, "elli": 50}
_EN_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
           "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50}
_TR_MON = ["ocak", "subat", "mart", "nisan", "mayis", "haziran", "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik"]
_EN_MON = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_TR_DAY = ["pazartesi", "sali", "carsamba", "persembe", "cuma", "cumartesi", "pazar"]   # index = date.weekday()
_EN_DAY = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_HALF = {"d": 0, "w": 3, "m": 15, "y": 182}   # tolerance (days) around a point "N units ago"

_TRN = r"(?:\d{1,3}|(?:(?:yirmi|otuz|kirk|elli)\s+)?(?:" + "|".join(k for k in _TR_NUM if k not in ("yirmi", "otuz", "kirk", "elli")) + r"))"
_ENN = r"(?:\d{1,3}|(?:(?:twenty|thirty|forty|fifty)[ -])?(?:" + "|".join(k for k in _EN_NUM if k not in ("a", "an", "twenty", "thirty", "forty", "fifty")) + r")|an?)"
_TR_SUF = r"(?:'?(?:nin|in|un|ndaki|ndeki|daki|deki|taki|teki|nda|nde|da|de|ta|te|dan|den|tan|ten|ki|ndan|nden|a|e|ya|ye))?"
_TR_UNIT = {"gun": "d", "hafta": "w", "ay": "m", "yil": "y", "sene": "y"}
_EN_UNIT = {"day": "d", "week": "w", "month": "m", "year": "y"}


def _num(s, table):
    s = s.strip()
    return int(s) if s.isdigit() else sum(table.get(w, 0) for w in re.split(r"[ -]", s))


def _addm(d, n):
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _month(y, m):
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def _month_back(ref, k):
    y, m = divmod(ref.year * 12 + ref.month - 1 - k, 12)
    return _month(y, m + 1)


def _week(ref, back):   # Monday..Sunday, `back` weeks before the one holding ref
    mon = ref - timedelta(days=ref.weekday() + 7 * back)
    return mon, mon + timedelta(days=6)


def _ago(ref, n, u):
    c = {"d": ref - timedelta(days=n), "w": ref - timedelta(weeks=n), "m": _addm(ref, -n), "y": _addm(ref, -12 * n)}[u]
    return c - timedelta(days=_HALF[u]), min(c + timedelta(days=_HALF[u]), ref)


def _span(ref, n, u):   # "last/past N units": from N units back to ref
    return {"d": ref - timedelta(days=n), "w": ref - timedelta(weeks=n), "m": _addm(ref, -n), "y": _addm(ref, -12 * n)}[u], ref


def _lastday(ref, wd):   # "last <weekday>": one date when the weekday is later in the week than ref, else both readings
    back = (ref.weekday() - wd) % 7 or 7
    d = ref - timedelta(days=back)
    return (d, d) if d < ref - timedelta(days=ref.weekday()) else (d - timedelta(days=7), d)


def _bare_day(ref, wd):   # weekday with no "last": most recent one up to ref and the one before it
    d = ref - timedelta(days=(ref.weekday() - wd) % 7)
    return d - timedelta(days=7), d


def _lastweekend(ref):
    sat = ref - timedelta(days=(ref.weekday() - 5) % 7 or 7)
    if ref.weekday() >= 5:   # on Sat/Sun "last weekend" may mean today's
        return sat, ref
    return sat, sat + timedelta(days=1)


def _dm(ref, day, mon, year=None):   # day+month, year inferred as the latest one not after ref
    try:
        if year:
            return (date(year, mon, day),) * 2
        d = date(ref.year, mon, day)
        d = d if d <= ref else date(ref.year - 1, mon, day)
        return d, d
    except ValueError:
        return None


def _mon_year(ref, mon, year):
    if year:
        return _month(year, mon)
    y = ref.year if mon <= ref.month else ref.year - 1
    return _month(y, mon)


def _tr_rules(t, ref):
    """Yield (start, end, since, until) for Turkish expressions in folded, lowercased text t."""
    mon = "(" + "|".join(_TR_MON) + ")"
    day = "(" + "|".join(_TR_DAY[:6]) + ")"
    for m in re.finditer(r"\bbugun\b", t):
        yield m.start(), m.end(), ref, ref
    for m in re.finditer(r"\bdun\b", t):
        yield m.start(), m.end(), ref - timedelta(1), ref - timedelta(1)
    for m in re.finditer(r"\b(?:evvelsi|evvelki|onceki)\s+gun\w{0,3}\b", t):
        yield m.start(), m.end(), ref - timedelta(2), ref - timedelta(2)
    for m in re.finditer(r"\b(gecen|onceki|bu)\s+(hafta|ay|yil|sene)" + _TR_SUF + r"(?:\s+sonu)?\b", t):
        k, u = m.group(1) != "bu", _TR_UNIT[m.group(2)]
        wk = m.group(0).endswith("sonu")
        if wk and u == "w" and k:
            s, e = _lastweekend(ref)
        elif wk:
            continue   # "bu hafta sonu" is usually ahead, skip
        elif u == "w":
            s, e = _week(ref, 1 if k else 0)
        elif u == "m":
            s, e = _month_back(ref, k)
        else:
            s, e = date(ref.year - k, 1, 1), date(ref.year - k, 12, 31)
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\b(" + _TRN + r")\s+(gun|hafta|ay|yil|sene)\s+(?:once|evvel)\b", t):
        s, e = _ago(ref, _num(m.group(1), _TR_NUM), _TR_UNIT[m.group(2)])
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\bson\s+(" + _TRN + r")\s+(gun|hafta|ay|yil|sene)\w{0,3}\b", t):
        s, e = _span(ref, _num(m.group(1), _TR_NUM), _TR_UNIT[m.group(2)])
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\b(?:(gecen)\s+)?" + day + r"\w{0,4}(?:\s+gun\w{0,3})?\b", t):
        w = m.group(2)
        if m.group(2) is None:
            continue
        tail = t[m.end(2):m.end()]
        if not re.fullmatch(r"(?:'?(?:den|dan|ten|tan|de|da|te|ta|ki|yi|yu|ye|ya|e|a|in|nin|i|u))?(?:\s+gun\w{0,3})?", tail):
            continue
        wd = _TR_DAY.index(w)
        s, e = _lastday(ref, wd) if m.group(1) else _bare_day(ref, wd)
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\bgecen\s+pazar\b|\bpazar\s+gun\w{0,3}\b", t):
        s, e = _lastday(ref, 6) if "gecen" in m.group(0) else _bare_day(ref, 6)
        yield m.start(), m.end(), s, e
    yr = r"(?:\s+((?:19|20)\d\d))?"
    for m in re.finditer(r"\b(?:(\d{1,2})\s+)?" + mon + _TR_SUF + r"(?:\s+ay\w{0,5})?" + yr + r"(?:'?(?:da|de|ta|te|dan|den|tan|ten|nin|in|a|e))?\b", t):
        d, mo, y = m.group(1), _TR_MON.index(m.group(2)) + 1, m.group(3)
        if d:
            r = _dm(ref, int(d), mo, int(y) if y else None)
            if r:
                yield m.start(), m.end(), *r
        else:
            yield m.start(), m.end(), *_mon_year(ref, mo, int(y) if y else None)
    for m in re.finditer(r"\b((?:19|20)\d\d)(?:'?(?:te|ta|de|da)\b|\s+(?:yil|sene)\w*)", t):
        yield m.start(), m.end(), date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31)


def _en_rules(q, ref):
    """Yield (start, end, since, until) for English expressions; q is the original text (month names are case-checked)."""
    t = q.lower()
    for m in re.finditer(r"\btoday\b", t):
        yield m.start(), m.end(), ref, ref
    for m in re.finditer(r"\byesterday\b", t):
        yield m.start(), m.end(), ref - timedelta(1), ref - timedelta(1)
    for m in re.finditer(r"\bthe day before yesterday\b", t):
        yield m.start(), m.end(), ref - timedelta(2), ref - timedelta(2)
    for m in re.finditer(r"\b(last|previous|this)\s+(week|month|year)\b(?!\s*'?s?\s+(?:to|and)\b)", t):
        k, u = m.group(1) != "this", _EN_UNIT[m.group(2)]
        if u == "w":
            s, e = _week(ref, int(k))
        elif u == "m":
            s, e = _month_back(ref, k)
        else:
            s, e = date(ref.year - k, 1, 1), date(ref.year - k, 12, 31)
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\b(?:last|past)\s+weekend\b", t):
        yield m.start(), m.end(), *_lastweekend(ref)
    for m in re.finditer(r"\b(" + _ENN + r"|a couple of)\s+(day|week|month|year)s?\s+ago\b", t):
        if m.group(1) == "a couple of":
            yield m.start(), m.end(), ref - timedelta(3), ref - timedelta(1)   # "a couple" = 2, sometimes 3
            continue
        s, e = _ago(ref, _num(m.group(1), _EN_NUM), _EN_UNIT[m.group(2)])
        yield m.start(), m.end(), s, e
    for m in re.finditer(r"\b(?:in the |over the |during the )?(?:last|past)\s+(" + _ENN + r"\s+)?(day|week|month|year)s?\b(?!\s+(?:of|to|and|before)\b)", t):
        if m.group(1) is None and m.group(0).startswith(("last", "past")) and m.group(2) in ("day",):
            continue
        n = _num(m.group(1), _EN_NUM) if m.group(1) else 1
        s, e = _span(ref, n, _EN_UNIT[m.group(2)])
        i = m.group(0).find(("last" if "last" in m.group(0) else "past"))
        yield m.start() + i, m.end(), s, e
    for m in re.finditer(r"\blast\s+(" + "|".join(_EN_DAY) + r")\b", t):
        yield m.start(), m.end(), *_lastday(ref, _EN_DAY.index(m.group(1)))
    for m in re.finditer(r"\bthe\s+(" + "|".join(_EN_DAY) + r")\s+(" + _ENN + r")\s+(week|month)s?\s+ago\b", t):
        # weekday inside an N-weeks-ago window: only valid when the window pins one calendar week
        if m.group(3) != "week":
            continue
        base = ref - timedelta(weeks=_num(m.group(2), _EN_NUM))
        d = base - timedelta(days=(base.weekday() - _EN_DAY.index(m.group(1))) % 7)
        yield m.start(), m.end(), d - timedelta(7), d + timedelta(7)
    pre = r"(?:in|during|of|since|until|from|through|between|early|late|mid|and|to|or|by|on)\s+(?:the\s+)?(?:(?:early|late|mid)[- ])?"
    mon = "|".join(_EN_MON)
    cap = r"(?-i:\b(?:January|February|March|April|June|July|August|September|October|November|December|May)\b)"
    for m in re.finditer(r"(?i:\b(?:" + pre + r")?)(?P<mon>" + cap + r"|(?i:" + mon + r"))(?:\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>(?:19|20)\d\d)\b)?|,?\s+(?P<y2>(?:19|20)\d\d)\b)?", q):
        name = m.group("mon").lower()
        prep = re.match(r"(?i)(?:in|during|of|since|until|from|through|between|early|late|mid|and|to|or|by|on)\b", m.group(0)) is not None
        has_num = m.group("d") or m.group("y2")
        if not (prep or has_num) or (name == "may" and m.group("mon") != "May") or (m.group("mon")[0].islower() and not prep and not has_num):
            continue
        if name == "march" and not (prep or has_num):
            continue
        mo = _EN_MON.index(name) + 1
        if m.group("d"):
            r = _dm(ref, int(m.group("d")), mo, int(m.group("y")) if m.group("y") else None)
            if r:
                yield m.start(), m.end(), *r
        else:
            y = m.group("y2")
            yield m.start(), m.end(), *_mon_year(ref, mo, int(y) if y else None)
    for m in re.finditer(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(" + mon + r")(?:,?\s+((?:19|20)\d\d))?\b", t):
        r = _dm(ref, int(m.group(1)), _EN_MON.index(m.group(2)) + 1, int(m.group(3)) if m.group(3) else None)
        if r:
            yield m.start(), m.end(), *r
    for m in re.finditer(r"\b(?:in|during|of|from)\s+((?:19|20)\d\d)\b(?!\s*[-/:.]?\d)", t):
        yield m.start(), m.end(), date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31)


def _open_ended(t, start, end, ref):
    """'since March' / 'pazartesiden beri': the range runs on to ref, extend it. Returns (new_start, new_end, until) or None."""
    if re.match(r"\s*(?:beri|itibaren)\b", t[end:]):
        return start, end + len(re.match(r"\s*(?:beri|itibaren)", t[end:]).group(0)), ref
    if re.match(r"since\b", t[start:end]) or re.search(r"\bsince\s+(?:the\s+)?$", t[:start]):
        return start, end, ref
    return None


def parse(query, ref_date):
    """See module docstring. ref_date: datetime.date (a datetime is accepted)."""
    ref = ref_date.date() if hasattr(ref_date, "date") else ref_date
    q = query.translate(_FOLD)   # 1:1 char map so offsets stay valid; only used for the Turkish rules
    t = q.lower()
    hits = []
    for rules, txt in ((_tr_rules, t), (_en_rules, query)):
        for s, e, a, b in rules(txt, ref):
            # "before today", "şubattan önce": a one-sided bound, not a closed range, so drop it (precision over coverage)
            one_sided = re.search(r"\b(?:before|after|until|till|prior to)\s+(?:the\s+)?$", query[:s].lower()) or re.match(r"\s*(?:once|sonra)\b", t[e:])
            if a and b and a <= b and not one_sided:
                ext = _open_ended(t, s, e, ref)
                if ext:
                    s, e, b = ext[0], ext[1], max(b, ref)
                hits.append((s, e, a, b))
    hits.sort(key=lambda h: (h[0] - h[1], h[0]))   # longest first, so a contained or overlapping shorter hit loses
    keep = []
    for h in hits:
        if all(h[1] <= k[0] or h[0] >= k[1] for k in keep):
            keep.append(h)
    if not keep:
        return None
    return (min(k[2] for k in keep), max(k[3] for k in keep), (min(k[0] for k in keep), max(k[1] for k in keep)))
