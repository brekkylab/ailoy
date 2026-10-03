// `Map.prototype.getOrInsert` and `getOrInsertComputed`.
//
// A stage-3 proposal that pdf.js 6 uses throughout, in both its main bundle and its
// worker. WebKit has not shipped it, so on macOS every page render fails with
// `getOrInsertComputed is not a function` — from inside a promise pdf.js swallows, which
// is why it presented as a blank page rather than as an error.
//
// Importing this module installs them. Guarded, so it does nothing once the engine has
// them and disappears as a difference rather than as a permanent fork; the semantics are
// the proposal's, which is also what pdf.js is written against.
//
// Side-effecting on purpose. That is what a polyfill is, and it is why this is imported
// for itself rather than called.

interface Upsert<K, V> {
  getOrInsert?(key: K, value: V): V;
  getOrInsertComputed?(key: K, callback: (key: K) => V): V;
}

const proto = Map.prototype as Map<unknown, unknown> & Upsert<unknown, unknown>;

if (typeof proto.getOrInsert !== "function") {
  proto.getOrInsert = function getOrInsert(key, value) {
    if (this.has(key)) return this.get(key);
    this.set(key, value);
    return value;
  };
}

if (typeof proto.getOrInsertComputed !== "function") {
  proto.getOrInsertComputed = function getOrInsertComputed(key, callback) {
    // `has` rather than a null check: a key whose stored value is `undefined` is present,
    // and recomputing it would call a callback the proposal says is called once.
    if (this.has(key)) return this.get(key);
    const value = callback(key);
    this.set(key, value);
    return value;
  };
}

export {};
