// Native capture panel for memrove grab (macOS). Python decides what can be saved and does the saving; this file only draws
// the panel and asks. Protocol, one JSON object per line: Python writes the model first, then answers each request this
// panel writes to stdout ({"op": "hint" | "create" | "save" | "cancel", ...}) with one reply line on stdin.
// Look: Panel's chrome (Claude-style warm dark, monochrome ivory accent, neutral selection), Panel's type scale (S = interface
// text size; body S, secondary S-1, caption S-2, composer S+4), Linear/Things-style property pills, native text editing.
// Two modes, one binary. `MemroveGrab` (no argument) is the pipe-driven panel above. `MemroveGrab agent --vault V --python P
// --script S --keycode K --mods M --label L` is the resident menu bar agent the LaunchAgent runs: an accessory app (no Dock
// icon, no app switcher entry) with a status item, the global hotkey (Carbon, no Accessibility permission) and, per press, one
// headless `python S window` that gathers context, spawns this binary in panel mode and saves. No Python window is ever shown.
// Build: swiftc -O -swift-version 5 memrove_v3_grab_panel.swift -o MemroveGrab
import AppKit
import Carbon.HIToolbox
import SwiftUI

// MARK: Model and bridge

struct Source: Codable, Hashable { let id: String; let symbol: String; let label: String; let title: String; let detail: String }
struct Group: Codable, Hashable { let name: String; var count: Int }
struct Model: Codable {
    let s: Int
    let palette: [String: String]
    let sources: [Source]
    let source: String?
    let clip: String?
    let includeClip: Bool
    let groups: [Group]
    let selected: [String]
    let hint: String
    let app: String?
}

enum Bridge {
    static func send(_ object: [String: Any]) -> [String: Any] {
        guard let data = try? JSONSerialization.data(withJSONObject: object), let line = String(data: data, encoding: .utf8) else { return [:] }
        FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
        guard let reply = readLine(strippingNewline: true), let replyData = reply.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: replyData) as? [String: Any] else { return [:] }
        return parsed
    }
}

func fold(_ text: String) -> String {
    // Turkish-aware lowercase (dotted/dotless i).
    text.replacingOccurrences(of: "İ", with: "i").replacingOccurrences(of: "I", with: "ı").lowercased(with: Locale(identifier: "tr"))
}

extension Color {
    init(hex: String) {
        var value: UInt64 = 0
        Scanner(string: hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))).scanHexInt64(&value)
        self.init(.sRGB, red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255, blue: Double(value & 0xFF) / 255)
    }
}

struct Theme {
    let bg, raised, line, text, secondary, tertiary, faint, accent, onAccent, ok, warn, error: Color
    let hover = Color(hex: "#F0EFE9").opacity(0.08)
    let selected = Color(hex: "#F0EFE9").opacity(0.11)
    let s: CGFloat
    init(_ m: Model) {
        let p = { (key: String, fallback: String) in Color(hex: m.palette[key] ?? fallback) }
        bg = p("bg", "#222221"); raised = p("card", "#292928"); line = p("line", "#302F2E"); text = p("text", "#F0EFE9")
        secondary = p("muted", "#A7A7A3"); tertiary = p("faint", "#8E8D8A"); faint = p("key_line", "#494846")
        accent = p("accent", "#F0EFE9"); onAccent = p("accent_text", "#222221")
        ok = p("ok", "#BAC761"); warn = p("warn", "#E0C560"); error = p("error", "#D08183")
        s = CGFloat(max(11, min(16, m.s)))
    }
    func font(_ offset: CGFloat, _ weight: Font.Weight = .regular) -> Font { .system(size: max(11, s + offset), weight: weight) }
}

@MainActor final class Store: ObservableObject {
    let model: Model
    let theme: Theme
    @Published var text = ""
    @Published var groups: [Group]
    @Published var selected: [String]
    @Published var hint: String
    @Published var source: String?
    @Published var includeClip: Bool
    @Published var open = false
    @Published var pickerShown = false
    @Published var status: (String, Color)? = nil
    @Published var saved = false
    var finish: () -> Void = {}

    init(_ model: Model) {
        self.model = model
        theme = Theme(model)
        groups = model.groups
        selected = model.selected
        hint = model.hint
        source = model.source
        includeClip = model.includeClip
    }

    var current: Source? { model.sources.first { $0.id == source } }
    var isThought: Bool { source == nil }
    var placeholder: String { isThought ? "Write what's on your mind…" : (hint.isEmpty ? "Why are you saving this?" : hint) }

    func toggle(_ name: String) {
        if let index = selected.firstIndex(where: { fold($0) == fold(name) }) { selected.remove(at: index) } else { selected.append(name) }
        refreshHint()
    }

    func refreshHint() {
        hint = Bridge.send(["op": "hint", "groups": selected])["hint"] as? String ?? ""
    }

    func create(_ raw: String) -> String? {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let reply = Bridge.send(["op": "create", "name": name])
        guard let stored = reply["name"] as? String else {
            status = ((reply["error"] as? String) ?? "Could not create the group", theme.error)
            return nil
        }
        if !groups.contains(where: { fold($0.name) == fold(stored) }) { groups.insert(Group(name: stored, count: 0), at: 0) }
        if !selected.contains(where: { fold($0) == fold(stored) }) { selected.append(stored) }
        refreshHint()
        return stored
    }

    /// The source Tab switches to: the next one in order, when there is more than one.
    var otherSource: Source? {
        let all = model.sources
        guard all.count > 1, let at = all.firstIndex(where: { $0.id == source }) else { return nil }
        return all[(at + 1) % all.count]
    }

    func cycleSource() {
        let ids = model.sources.map(\.id)
        guard ids.count > 1, let at = ids.firstIndex(where: { $0 == source }) else { return }
        source = ids[(at + 1) % ids.count]
    }

    func save() {
        guard !saved else { return }
        if isThought && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            status = ("Write something first", theme.warn)
            return
        }
        let reply = Bridge.send(["op": "save", "text": text, "groups": selected, "open": open, "source": source ?? NSNull(),
                                 "includeClip": includeClip])
        if let error = reply["error"] as? String {
            status = ("Save_failed: " + error, theme.error)
            return
        }
        saved = true
        status = (isThought ? "Thought saved" : "Saved", theme.ok)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { self.finish() }
    }

    func cancel() {
        guard !saved else { return }
        saved = true
        _ = Bridge.send(["op": "cancel"])
        finish()
    }
}

// MARK: Pieces

struct Pill<Label: View>: View {
    let theme: Theme
    var on = false
    let action: () -> Void
    @ViewBuilder let label: () -> Label
    @State private var hover = false
    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) { label() }
                .font(theme.font(-1))
                .foregroundStyle(on ? theme.text : theme.secondary)
                .padding(.horizontal, 9)
                .frame(height: theme.s + 11)
                .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(on ? theme.selected : (hover ? theme.hover : .clear)))
                .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).strokeBorder(on ? theme.faint : theme.line, lineWidth: 1))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .focusable(false)
        .onHover { hover = $0 }
    }
}

struct Keycap: View {
    let theme: Theme
    let text: String
    var body: some View {
        Text(text).font(theme.font(-2)).foregroundStyle(theme.tertiary)
            .padding(.horizontal, 5).frame(minWidth: theme.s + 5, minHeight: theme.s + 5)
            .background(RoundedRectangle(cornerRadius: 4).fill(theme.line))
    }
}

struct SourceChip: View {
    @ObservedObject var store: Store
    let item: Source
    @State private var hover = false
    @State private var swapHover = false
    var body: some View {
        let t = store.theme
        HStack(spacing: 10) {
            Image(systemName: item.symbol).font(t.font(0)).foregroundStyle(t.secondary).frame(width: t.s + 9)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(t.font(0)).foregroundStyle(t.text).lineLimit(1).truncationMode(.tail)
                Text([item.label, item.detail].filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(t.font(-1)).foregroundStyle(t.secondary).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 8)
            if let other = store.otherSource {
                Button { store.cycleSource() } label: {
                    HStack(spacing: 5) {
                        Image(systemName: other.symbol).font(t.font(-2))
                        Text(other.label).font(t.font(-1))
                        Keycap(theme: t, text: "⇥")
                    }
                    .foregroundStyle(t.secondary).padding(.leading, 8).padding(.trailing, 3).frame(height: t.s + 11)
                    .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(swapHover ? t.hover : .clear))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain).focusable(false).onHover { swapHover = $0 }
                .help("Save \(other.label.lowercased()) instead  ⇥").accessibilityLabel("Source: \(other.label)")
            }
            Button { store.source = nil } label: { Image(systemName: "xmark").font(t.font(-2, .medium)) }
                .buttonStyle(.plain).foregroundStyle(hover ? t.text : t.tertiary)
                .help("Remove the source, save as a thought").accessibilityLabel("Remove source")
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(RoundedRectangle(cornerRadius: 10).fill(t.raised))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(t.line, lineWidth: 1))
        .onHover { hover = $0 }
    }
}

// MARK: Group picker (a native popover: search, checkmark rows with counts, inline create)

struct GroupPicker: View {
    @ObservedObject var store: Store
    @State private var query = ""
    @State private var focusIndex = 0
    @State private var creating = false
    @State private var newName = ""
    @FocusState private var searchFocused: Bool
    @FocusState private var nameFocused: Bool

    struct Row: Hashable { let name: String; let count: Int; let create: Bool }

    var rows: [Row] {
        let q = fold(query.trimmingCharacters(in: .whitespaces))
        let matches = store.groups.filter { q.isEmpty || fold($0.name).contains(q) }.map { Row(name: $0.name, count: $0.count, create: false) }
        let exact = store.groups.contains { fold($0.name) == q }
        return (!q.isEmpty && !exact ? [Row(name: query.trimmingCharacters(in: .whitespaces), count: 0, create: true)] : []) + matches
    }

    func activate(_ row: Row) {
        if row.create { _ = store.create(row.name); query = "" } else { store.toggle(row.name) }
    }

    var body: some View {
        let t = store.theme
        let rowH = t.s + 15
        VStack(spacing: 0) {
            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass").font(t.font(-1)).foregroundStyle(t.tertiary)
                TextField("Search or create a group", text: $query)
                    .textFieldStyle(.plain).font(t.font(0)).foregroundStyle(t.text).focused($searchFocused)
                    .onSubmit { if let row = rows[safe: focusIndex] { activate(row) } }
                    .onKeyPress(.downArrow) { focusIndex = min(focusIndex + 1, max(rows.count - 1, 0)); return .handled }
                    .onKeyPress(.upArrow) { focusIndex = max(focusIndex - 1, 0); return .handled }
                    .onChange(of: query) { focusIndex = 0 }
            }
            .padding(.horizontal, 12).frame(height: rowH + 4)
            Divider().overlay(t.line)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 1) {
                        ForEach(Array(rows.enumerated()), id: \.element) { index, row in
                            PickerRow(store: store, row: row, focused: index == focusIndex, height: rowH) { activate(row) }
                                .id(index)
                        }
                        if rows.isEmpty && !creating {
                            Text("No groups yet").font(t.font(-1)).foregroundStyle(t.secondary).frame(height: rowH)
                        }
                    }
                    .padding(5)
                }
                .frame(height: min(CGFloat(max(rows.count, 1)) * (rowH + 1) + 10, 8 * (rowH + 1) + 10))
                .onChange(of: focusIndex) { proxy.scrollTo(focusIndex) }
            }
            Divider().overlay(t.line)
            if creating {
                HStack(spacing: 8) {
                    TextField("New group name", text: $newName)
                        .textFieldStyle(.plain).font(t.font(0)).foregroundStyle(t.text).focused($nameFocused)
                        .onSubmit(submitNew)
                        .onExitCommand { creating = false; searchFocused = true }
                    Button("Create", action: submitNew)
                        .buttonStyle(.plain).font(t.font(-1, .medium)).foregroundStyle(t.onAccent)
                        .padding(.horizontal, 10).frame(height: t.s + 9)
                        .background(Capsule().fill(t.accent))
                        .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .padding(.horizontal, 12).frame(height: rowH + 6)
            } else {
                Button { creating = true; newName = query; DispatchQueue.main.async { nameFocused = true } } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "plus").font(t.font(-1, .medium)).frame(width: t.s)
                        Text("New group…").font(t.font(0))
                        Spacer()
                    }
                    .foregroundStyle(t.secondary).padding(.horizontal, 12).frame(height: rowH + 6).contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(width: 18 * t.s + 46)
        .background(t.bg)
        .onAppear { DispatchQueue.main.async { searchFocused = true } }
    }

    func submitNew() {
        if store.create(newName) != nil { newName = ""; creating = false; query = ""; searchFocused = true }
    }
}

struct PickerRow: View {
    @ObservedObject var store: Store
    let row: GroupPicker.Row
    let focused: Bool
    let height: CGFloat
    let action: () -> Void
    @State private var hover = false
    var body: some View {
        let t = store.theme
        let on = store.selected.contains { fold($0) == fold(row.name) }
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: row.create ? "plus" : "checkmark").font(t.font(-1, .medium))
                    .foregroundStyle(row.create ? t.secondary : t.text).opacity(row.create || on ? 1 : 0).frame(width: t.s)
                Text(row.create ? "“\(row.name)” create" : row.name).font(t.font(0)).foregroundStyle(t.text).lineLimit(1)
                Spacer()
                if row.create { Keycap(theme: t, text: "↵") } else { Text("\(row.count)").font(t.font(-1)).foregroundStyle(t.secondary).monospacedDigit() }
            }
            .padding(.horizontal, 8).frame(height: height)
            .background(RoundedRectangle(cornerRadius: 6).fill(focused || hover ? t.selected : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .accessibilityLabel(row.create ? "Create group \(row.name)" : row.name)
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}

extension Array { subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil } }

// MARK: Panel

struct PanelView: View {
    @ObservedObject var store: Store
    @FocusState private var composerFocused: Bool

    var body: some View {
        let t = store.theme
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 14) {
                TextField(store.placeholder, text: $store.text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(t.font(4))
                    .foregroundStyle(t.text)
                    .lineLimit(1...6)
                    .focused($composerFocused)
                    .disabled(store.saved)
                    .onKeyPress(.tab, phases: .down) { press in
                        guard store.otherSource != nil, press.modifiers.isEmpty else { return .ignored }
                        store.cycleSource()
                        return .handled
                    }
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) || press.modifiers.contains(.option) { return .ignored }
                        store.save()
                        return .handled
                    }
                if let item = store.current { SourceChip(store: store, item: item) }
                HStack(spacing: 7) {
                    Pill(theme: t, on: !store.selected.isEmpty, action: { store.pickerShown.toggle() }) {
                        Image(systemName: store.selected.isEmpty ? "plus" : "tag")
                        Text(store.selected.isEmpty ? "Group" : store.selected.joined(separator: ", ")).lineLimit(1)
                    }
                    .keyboardShortcut("g", modifiers: .command)
                    .help("Groups  ⌘G")
                    .popover(isPresented: $store.pickerShown, arrowEdge: .bottom) { GroupPicker(store: store) }
                    Pill(theme: t, on: store.open, action: { store.open.toggle() }) {
                        Image(systemName: store.open ? "largecircle.fill.circle" : "circle")
                        Text("Open loop")
                    }
                    .keyboardShortcut("t", modifiers: .command)
                    .help("Mark this as an open loop  ⌘T")
                    if store.model.clip != nil && store.current.map({ $0.id != "clip" }) == true {
                        Pill(theme: t, on: store.includeClip, action: { store.includeClip.toggle() }) {
                            Image(systemName: "doc.on.clipboard")
                            Text("Clipboard text")
                        }
                        .help(store.model.clip ?? "")
                    }
                    Spacer(minLength: 0)
                }
            }
            .padding(.horizontal, 20).padding(.top, 18).padding(.bottom, 16)
            Rectangle().fill(t.line).frame(height: 1)
            HStack(spacing: 10) {
                if let (message, color) = store.status {
                    HStack(spacing: 5) {
                        if color == t.ok { Image(systemName: "checkmark") }
                        Text(message).lineLimit(1).truncationMode(.tail)
                    }
                    .font(t.font(-1)).foregroundStyle(color)
                } else {
                    Text(footerNote).font(t.font(-1)).foregroundStyle(t.secondary).lineLimit(1).truncationMode(.tail)
                }
                Spacer(minLength: 12)
                Button(action: store.save) {
                    HStack(spacing: 6) {
                        Text("Save").font(t.font(-1, .medium))
                        Image(systemName: "return").font(t.font(-2, .medium))
                    }
                    .foregroundStyle(t.onAccent).padding(.horizontal, 11).frame(height: t.s + 11)
                    .background(Capsule().fill(t.accent))
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain).disabled(store.saved).accessibilityLabel("Save")
            }
            .padding(.horizontal, 20).frame(height: t.s + 31)
        }
        .frame(width: 44 * t.s + 48)
        .background(RoundedRectangle(cornerRadius: 14).fill(t.bg))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(t.line, lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .tint(t.accent)
        .onExitCommand { store.cancel() }
        .onAppear { DispatchQueue.main.async { composerFocused = true } }
    }

    var footerNote: String {
        let from = store.model.app.map { "\($0)'dan" } ?? ""
        if store.isThought { return "Thought · stored as you typed it" }
        return from
    }
}

// MARK: Window

final class KeyPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

@MainActor func makePanel(_ store: Store, offscreen: Bool) -> KeyPanel {
    let host = NSHostingView(rootView: PanelView(store: store))
    host.layoutSubtreeIfNeeded()
    let size = host.fittingSize
    let panel = KeyPanel(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
                         backing: .buffered, defer: false)
    panel.contentView = host
    panel.isOpaque = false
    panel.backgroundColor = .clear
    panel.hasShadow = true
    panel.level = .floating
    panel.isMovableByWindowBackground = true
    panel.hidesOnDeactivate = false
    panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
    panel.appearance = NSAppearance(named: .darkAqua)
    if offscreen {
        panel.setFrameOrigin(NSPoint(x: -30000, y: -30000))
    } else if let screen = NSScreen.main?.visibleFrame {
        panel.setFrameTopLeftPoint(NSPoint(x: screen.midX - size.width / 2, y: screen.maxY - screen.height * 0.2))
    }
    return panel
}

@MainActor func snapshot(_ view: NSView, to path: String) {
    view.layoutSubtreeIfNeeded()
    guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
    view.cacheDisplay(in: view.bounds, to: rep)
    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
}

final class Delegate: NSObject, NSApplicationDelegate {
    var panel: KeyPanel?
    var store: Store?

    func applicationDidFinishLaunching(_ note: Notification) {
        MainActor.assumeIsolated { start() }
    }

    @MainActor func start() {
        guard let line = readLine(strippingNewline: true), let data = line.data(using: .utf8),
              let model = try? JSONDecoder().decode(Model.self, from: data) else { exit(2) }
        let store = Store(model)
        self.store = store
        let env = ProcessInfo.processInfo.environment
        if let path = env["GRAB_SNAPSHOT"] {  // render off-screen, never on the owner's screen, then exit
            if let typed = env["GRAB_SNAPSHOT_TEXT"] { store.text = typed }
            if env["GRAB_SNAPSHOT_OPEN"] != nil { store.open = true }
            let panel = makePanel(store, offscreen: true)
            panel.orderBack(nil)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                snapshot(panel.contentView!, to: path)
                let picker = NSHostingView(rootView: GroupPicker(store: store).environment(\.colorScheme, .dark))
                picker.frame = NSRect(origin: .zero, size: picker.fittingSize)
                let holder = NSWindow(contentRect: picker.frame, styleMask: [.borderless], backing: .buffered, defer: false)
                holder.contentView = picker
                holder.setFrameOrigin(NSPoint(x: -30000, y: -29000))
                holder.orderBack(nil)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    snapshot(picker, to: path.replacingOccurrences(of: ".png", with: "-picker.png"))
                    exit(0)
                }
            }
            return
        }
        let panel = makePanel(store, offscreen: false)
        self.panel = panel
        if env["GRAB_GROUPS"] != nil {  // menu bar "Groups…": open with the group picker showing
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { MainActor.assumeIsolated { store.pickerShown = true } }
        }
        store.finish = { exit(0) }
        NotificationCenter.default.addObserver(forName: NSWindow.didResignKeyNotification, object: panel, queue: .main) { _ in
            // A click elsewhere closes the panel; focus moving into our own popover does not.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                MainActor.assumeIsolated {
                    if NSApp.keyWindow == nil && !store.pickerShown { store.cancel() }
                }
            }
        }
        let reduce = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        panel.alphaValue = reduce ? 1 : 0
        panel.makeKeyAndOrderFront(nil)
        if !reduce { NSAnimationContext.runAnimationGroup { $0.duration = 0.12; panel.animator().alphaValue = 1 } }
    }
}

// MARK: Menu bar agent

final class Agent: NSObject, NSApplicationDelegate {
    let options: [String: String]
    var item: NSStatusItem?
    var child: Process?
    var hotKey: EventHotKeyRef?

    init(_ arguments: [String]) {
        var found: [String: String] = [:]
        var index = 0
        while index + 1 < arguments.count {
            if arguments[index].hasPrefix("--") { found[String(arguments[index].dropFirst(2))] = arguments[index + 1]; index += 2 } else { index += 1 }
        }
        options = found
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        let label = options["label"] ?? "⌃⌥B"
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            let image = NSImage(systemSymbolName: "tray.and.arrow.down", accessibilityDescription: "memrove Grab")
            image?.isTemplate = true
            button.image = image
            button.toolTip = "memrove Grab " + label
        }
        let menu = NSMenu()
        menu.addItem(withTitle: "memrove Grab  " + label, action: #selector(capture), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Groups…", action: #selector(groups), keyEquivalent: "").target = self
        let registered = registerHotKey()
        if !registered {
            let warning = menu.insertItem(withTitle: "Shortcut is registered in another app", action: nil, keyEquivalent: "", at: 0)
            warning.isEnabled = false
            item.button?.toolTip = "memrove Grab: shortcut is registered in another app"
        }
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "")
        item.menu = menu
        self.item = item
    }

    func registerHotKey() -> Bool {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let handler: EventHandlerUPP = { _, _, data in
            guard let data = data else { return noErr }
            let agent = Unmanaged<Agent>.fromOpaque(data).takeUnretainedValue()
            DispatchQueue.main.async { agent.run(groups: false) }
            return noErr
        }
        let target = GetApplicationEventTarget()
        guard InstallEventHandler(target, handler, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), nil) == noErr else { return false }
        let signature = "BYKL".utf8.reduce(OSType(0)) { $0 << 8 | OSType($1) }
        let status = RegisterEventHotKey(UInt32(options["keycode"] ?? "11") ?? 11, UInt32(options["mods"] ?? "6144") ?? 6144,
                                         EventHotKeyID(signature: signature, id: 1), target, 0, &hotKey)
        return status == noErr
    }

    @objc func capture() { run(groups: false) }
    @objc func groups() { run(groups: true) }

    /// One headless Python run per press; a second press while the panel is up is ignored.
    func run(groups: Bool) {
        if let child = child, child.isRunning { return }
        guard let python = options["python"], let runner = options["script"], let vault = options["vault"] else { return }
        let vaultCopy = vault + "/.claude/scripts/" + (runner as NSString).lastPathComponent  // newest copy: updates need no reinstall
        let script = FileManager.default.fileExists(atPath: vaultCopy) ? vaultCopy : runner
        var context: [String: String] = [:]
        if let front = NSWorkspace.shared.frontmostApplication {
            if let name = front.localizedName { context["app"] = name }
            if let bundle = front.bundleIdentifier { context["bundle"] = bundle }
        }
        let json = (try? JSONSerialization.data(withJSONObject: context)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = [script, "window", "--vault", vault, "--context", json]
        var environment = ProcessInfo.processInfo.environment
        if groups { environment["GRAB_GROUPS"] = "1" }
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        child = process
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)  // before anything else: no Dock icon, no app switcher entry
let agent = CommandLine.arguments.dropFirst().first == "agent" ? Agent(Array(CommandLine.arguments.dropFirst(2))) : nil
let delegate = Delegate()
app.delegate = agent ?? delegate
app.run()
