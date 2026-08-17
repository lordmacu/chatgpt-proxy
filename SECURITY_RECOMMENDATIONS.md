# Security Recommendations for ChatGPT Android Anonymous API

**To:** OpenAI / ChatGPT Platform Engineering  
**From:** Reverse engineering of the ChatGPT Android application (academic / educational research)  
**Repository:** https://github.com/lordmacu/chatgpt-proxy  
**Date:** 2026-08-16  

---

## Context

This document describes the attack surface discovered while reverse-engineering the ChatGPT Android APK and building a working open-source proxy that replicates the anonymous (unauthenticated) API flow. The proxy is fully functional: it speaks to the real ChatGPT backend at `https://android.chat.openai.com`, streams conversations, and rotates device identities to bypass per-device rate limits.

The goal of this document is not to embarrass but to be useful. Every technique described here is already public knowledge among mobile security researchers; publishing it together with mitigations gives your team a consolidated reference.

---

## 1. The Anonymous Flow Is Too Open

### What was found

The ChatGPT Android app exposes a fully functional anonymous endpoint:

```
POST https://android.chat.openai.com/backend-anon/f/conversation
```

The only required headers are:
- `oai-device-id: <any UUID v4>` — the sole "identity" for a session
- `oai-client-version: <static version string extracted from the APK>`
- Standard Android HTTP headers (User-Agent, Accept, etc.)

No authentication token, API key, or proof-of-device is validated server-side. A fresh UUID is immediately accepted and assigned a message quota. When the quota is exhausted, generating a new UUID restores the full quota in milliseconds. This makes the per-device limit entirely symbolic.

### Recommendations

**1a. Require server-verified device attestation before assigning a quota.**  
The [Play Integrity API](https://developer.android.com/google/play/integrity) produces a signed, server-verifiable token that proves the request originated from the genuine ChatGPT app on an un-rooted device. The backend should:
1. Require an Integrity token on the first request from any `oai-device-id`.
2. Verify the token against Google's Integrity Verdict API server-side.
3. Reject `MEETS_VIRTUAL_INTEGRITY` or `UNEVALUATED` verdicts.
4. Tie the quota to the verified token's `requestDetails.requestHash`, not to the raw device ID.

This alone breaks UUID rotation — a new UUID requires a fresh attestation token, which requires a real device.

**1b. Rate-limit at the network / IP layer, not only by device ID.**  
Device IDs are trivially rotated. The server should also enforce:
- Requests per IP per minute (accounting for CGNAT / mobile carriers via careful thresholds)
- Global rate limits per Play Integrity certificate hash (each app signing key has a certificate)
- Suspicious patterns: N distinct device IDs from the same IP within T minutes → soft/hard block

---

## 2. No Certificate Pinning (or Pinning That Can Be Bypassed)

### What was found

HTTP traffic was captured without any special effort. Standard MITM proxies (mitmproxy, Burp Suite) intercept the TLS connection without triggering any pin-validation failure. Either the app does not implement pinning or the current implementation does not enforce it in production builds.

### Recommendations

**2a. Implement multi-pin certificate pinning for production builds.**  
Pin at least two certificates (primary + backup) using the [OkHttp `CertificatePinner`](https://square.github.io/okhttp/4.x/okhttp/okhttp3/-certificate-pinner/) or equivalent:

```kotlin
val pinner = CertificatePinner.Builder()
    .add("android.chat.openai.com", "sha256/<primary-leaf-spki>")
    .add("android.chat.openai.com", "sha256/<backup-intermediate-spki>")
    .build()
```

Pin the SPKI (Subject Public Key Info) of the leaf certificate and at least one intermediate — leaf-only pins break on every cert rotation; intermediate-only pins are too broad.

**2b. Use network security config for defence-in-depth.**  
Android's `res/xml/network_security_config.xml` provides OS-enforced pinning independent of the app's HTTP client. Add it as a second layer:

```xml
<domain-config>
  <domain includeSubdomains="false">android.chat.openai.com</domain>
  <pin-set expiration="2027-01-01">
    <pin digest="SHA-256">primary_spki_base64==</pin>
    <pin digest="SHA-256">backup_spki_base64==</pin>
  </pin-set>
</domain-config>
```

**2c. Detect pinning bypass tools at runtime.**  
Libraries like [TrustKit](https://github.com/datatheorem/TrustKit-Android) optionally report pin failures to your telemetry pipeline before refusing the connection, giving you visibility into active MITM attempts.

---

## 3. Static Secrets Embedded in the APK

### What was found

The following values were extracted directly from the decompiled APK (SMALI / ProGuard output):

- `oai-client-version` header value
- Sentinel header structure (field names and format)
- API base URL (`android.chat.openai.com`)
- SSE event field names and encoding conventions (Unicode PUA markers ``–``)

None of these are cryptographically sensitive on their own, but together they are everything needed to construct a valid request.

### Recommendations

**3a. Don't treat static strings as secrets, but reduce their extractability.**  
Native code (via the Android NDK / JNI) is significantly harder to reverse-engineer than Java/Kotlin bytecode, even after ProGuard. Move the construction of critical request parameters (base URLs, header names, client version) into a native library. Combine with:
- String encryption at rest in the `.so` (decrypt at runtime, avoid keeping the key adjacent to the ciphertext)
- Anti-debug checks (detect `ptrace`, Frida, `ro.debuggable`, emulator fingerprints) to refuse to decrypt in instrumented environments

**3b. Rotate the client version regularly and reject old versions server-side.**  
The `oai-client-version` header is currently honoured indefinitely once extracted. Add a maximum allowed age for client versions and force-update older clients. This raises the maintenance cost for any proxy that hardcodes the version string.

**3c. Add a short-lived request nonce.**  
Include a server-issued nonce (fetched once per app session via an authenticated pre-flight) in the conversation request body. The nonce is single-use and expires in seconds. Without it the backend rejects the request. An attacker who replays a captured nonce finds it already consumed; one who doesn't have a nonce cannot start a session.

---

## 4. Predictable Device Identity

### What was found

`oai-device-id` is a client-generated UUID v4. The server accepts any syntactically valid UUID without verifying that it corresponds to a real device. Because UUID v4 is random, the client can generate unlimited "devices" programmatically.

### Recommendations

**4a. Derive the device ID from hardware-backed Android identity.**  
Use `Settings.Secure.ANDROID_ID` (unique per app/device/user combination since Android 8) as the seed, combined with the Play Integrity `deviceIntegrity` token. Store the resulting ID in the `EncryptedSharedPreferences` backed by the Android Keystore — it cannot be read on rooted devices without breaking the OS integrity that Play Integrity already checks.

**4b. Sign each request with a device-specific key.**  
Provision an asymmetric key pair in the Android Keystore (`KeyPairGenerator` with `StrongBoxBacked = true` where available). Sign a request hash with the private key and include the signature as a header. The backend verifies the signature against the public key registered during first-run attestation. The private key never leaves the secure element — it cannot be extracted and cannot be replicated on a non-device environment.

---

## 5. Absence of Behavioural Analysis

### What was found

The backend applies no visible behavioural heuristics. Requests from a programmatic client (no browser fingerprint, linear timing, no user-agent variation, no idle time between turns) are treated identically to requests from a real user. There is no CAPTCHA challenge, no secondary verification, and no apparent anomaly scoring.

### Recommendations

**5a. Collect passive signals and score them server-side.**  
- Time between user message and next API call (humans are slower)  
- Conversation turn rate (humans don't sustain 10 turns/minute)  
- Message length distribution (LLM-chained prompts have distinct statistical profiles)  
- Device model / OS version / locale consistency (a proxy typically uses a default or randomised value)

Score these signals per `oai-device-id` over a rolling window. Scores above a threshold trigger increasing friction (slower responses, reduced quota, CAPTCHA challenge on a future turn).

**5b. Correlate device IDs by network fingerprint.**  
TLS fingerprinting (JA3/JA4), TCP stack fingerprinting, and HTTP/2 SETTINGS frames are highly correlated with real Android devices vs. Python `httpx` clients. A TLS fingerprint mismatch between the declared `User-Agent` (Android Chrome / OkHttp) and the observed handshake is a strong signal of a proxy.

---

## 6. SSE Stream Is Fully Transparent

### What was found

The server-sent event format is readable plaintext. Field semantics were reverse-engineered from a single captured conversation:

- `event: delta` / `event: message` carry incremental text chunks
- `event: error` signals quota exhaustion or server error
- Unicode PUA range (``–``) delimits structured widgets (citations, Canvases, tools)
- Plain JSON objects with `"detail"` key signal API errors

### Recommendations

**6a. Encrypt or authenticate the stream for anonymous sessions (optional).**  
This is a harder trade-off because SSE is processed client-side in a WebView. An AEAD-encrypted stream (with the key negotiated during the pre-flight attestation step from §3c) would prevent proxies from parsing the content without having first completed attestation. The overhead is low; the friction for an attacker is high.

**6b. Vary delimiter conventions across client versions.**  
If the PUA marker values or event field names change between client versions, a proxy built against version N breaks silently on version N+1. Combine with §3b (short client version lifetime) for a regular rotation cadence.

---

## 7. Suggested Prioritisation

| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| 🔴 Critical | Play Integrity server-side verification (§1a) | Medium | Breaks UUID rotation entirely |
| 🔴 Critical | Certificate pinning enforcement (§2a–b) | Low | Blocks trivial MITM capture |
| 🟠 High | Device key signing via Android Keystore (§4b) | High | Makes request forgery infeasible |
| 🟠 High | IP + pattern rate limiting (§1b) | Low | Raises cost of large-scale abuse |
| 🟡 Medium | Native code for critical parameters (§3a) | Medium | Slows static extraction |
| 🟡 Medium | Short-lived request nonce (§3c) | Medium | Prevents replay attacks |
| 🟢 Low | Behavioural scoring (§5a–b) | High | Catches sophisticated proxies |
| 🟢 Low | Stream variation across versions (§6b) | Low | Raises maintenance cost for proxies |

---

## 8. What This Proxy Does NOT Exploit

For completeness:

- **Authentication tokens** — not stored, not accessed. This proxy uses only the anonymous flow; it does not touch any user account or OAuth token.
- **ChatGPT Desktop** — the desktop app was analysed and does not expose an anonymous endpoint. It loads `chatgpt.com` in a WebView and requires a full authenticated session.
- **Backend infrastructure** — no SQL injection, no path traversal, no privilege escalation. The proxy calls only the documented (if unofficial) public-facing API.

---

## Closing Note

The anonymous API flow is a valuable feature for users who want to try ChatGPT without signing in. The goal of these recommendations is not to remove it but to make it resistant to automated abuse while keeping the experience smooth for real users. The highest-leverage change — Play Integrity verification tied to the device ID — would close the UUID-rotation loophole entirely with minimal friction for legitimate users (Play Integrity checks happen silently in the background on non-rooted devices).

This document and the accompanying source code are published in the hope that they accelerate the hardening of this surface before it becomes a larger vector for abuse.
