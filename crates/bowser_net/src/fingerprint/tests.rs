//! Spot checks of vendored profiles against the wreq-util source data.

use super::*;

fn chrome_131() -> &'static Profile {
    profile("chrome_131").expect("chrome_131 must exist")
}

#[test]
fn chrome_131_tls() {
    let tls = chrome_131().tls;
    assert_eq!(
        tls.ciphers,
        "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:\
         TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:\
         TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:\
         TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:\
         TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:\
         TLS_RSA_WITH_AES_128_GCM_SHA256:TLS_RSA_WITH_AES_256_GCM_SHA384:\
         TLS_RSA_WITH_AES_128_CBC_SHA:TLS_RSA_WITH_AES_256_CBC_SHA"
    );
    assert_eq!(tls.curves, "X25519MLKEM768:X25519:P-256:P-384");
    assert_eq!(
        tls.sigalgs,
        "ecdsa_secp256r1_sha256:rsa_pss_rsae_sha256:rsa_pkcs1_sha256:\
         ecdsa_secp384r1_sha384:rsa_pss_rsae_sha384:rsa_pkcs1_sha384:\
         rsa_pss_rsae_sha512:rsa_pkcs1_sha512"
    );
    assert_eq!(tls.alpn, &["h2", "http/1.1"]);
    assert!(tls.grease);
    assert!(tls.permute_extensions);
    assert!(tls.ech_grease);
    assert!(tls.pre_shared_key);
    assert!(!tls.psk_skip_session_tickets);
    assert!(tls.psk_dhe_ke);
    assert!(tls.session_ticket);
    assert!(tls.ocsp_stapling);
    assert!(tls.signed_cert_timestamps);
    assert_eq!(tls.alps, "h2");
    assert!(!tls.alps_new_codepoint);
    assert_eq!(tls.cert_compressors, "brotli");
    assert!(!tls.sigalg_grease);
    assert!(!tls.trust_anchors);
    assert_eq!(tls.min_tls, TlsVersion::Tls12);
    assert_eq!(tls.max_tls, TlsVersion::Tls13);
    assert!(tls.key_shares.is_empty());
    assert_eq!(tls.delegated_credentials, "");
    assert_eq!(tls.record_size_limit, 0);
    assert!(tls.extension_order.is_empty());
}

#[test]
fn chrome_131_http2() {
    let h2 = chrome_131().http2;
    assert_eq!(h2.initial_window_size, Some(6291456));
    assert_eq!(h2.initial_connection_window_size, Some(15728640));
    assert_eq!(h2.max_header_list_size, Some(262144));
    assert_eq!(h2.header_table_size, Some(65536));
    assert_eq!(h2.max_concurrent_streams, None);
    assert_eq!(h2.max_frame_size, None);
    assert_eq!(h2.enable_push, Some(false));
    assert!(!h2.enable_connect_protocol);
    assert!(!h2.no_rfc7540_priorities);
    assert_eq!(h2.initial_stream_id, None);
    let dep = h2.stream_dependency.expect("chrome dep");
    assert_eq!((dep.parent, dep.weight, dep.exclusive), (0, 255, true));
    assert!(h2.priorities.is_empty());
    assert_eq!(
        h2.settings_order,
        &[
            "header_table_size",
            "enable_push",
            "max_concurrent_streams",
            "initial_window_size",
            "max_frame_size",
            "max_header_list_size",
            "enable_connect_protocol",
            "no_rfc7540_priorities"
        ]
    );
    assert_eq!(h2.pseudo_order, &["method", "authority", "scheme", "path"]);
}

#[test]
fn chrome_131_headers_and_ua() {
    let p = chrome_131();
    assert_eq!(p.header_set, HeaderSet::ChromeZstdPriority);
    assert_eq!(
        p.sec_ch_ua().as_deref(),
        Some("\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"")
    );
    assert_eq!(
        p.user_agent(Platform::Windows).as_deref(),
        Some(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    );
    assert_eq!(
        p.user_agent(Platform::MacOS).as_deref(),
        Some(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    );
    let tpl = header_template(p.header_set);
    assert_eq!(tpl.first().map(|f| f.name), Some("sec-ch-ua"));
    assert_eq!(tpl.last().map(|f| f.name), Some("priority"));
    let ua_field = tpl.iter().find(|f| f.name == "user-agent").unwrap();
    assert_eq!(
        p.header_value(Platform::Windows, ua_field).as_deref(),
        p.user_agent(Platform::Windows).as_deref()
    );
    let enc = tpl.iter().find(|f| f.name == "accept-encoding").unwrap();
    assert_eq!(
        p.header_value(Platform::Windows, enc).as_deref(),
        Some("gzip, deflate, br, zstd")
    );
    let pri = tpl.iter().find(|f| f.name == "priority").unwrap();
    assert_eq!(
        p.header_value(Platform::Windows, pri).as_deref(),
        Some("u=0, i")
    );
    let mobile = tpl.iter().find(|f| f.name == "sec-ch-ua-mobile").unwrap();
    assert_eq!(
        p.header_value(Platform::Android, mobile).as_deref(),
        Some("?1")
    );
    assert_eq!(
        p.header_value(Platform::MacOS, mobile).as_deref(),
        Some("?0")
    );
}

fn firefox_135() -> &'static Profile {
    profile("firefox_135").expect("firefox_135 must exist")
}

#[test]
fn firefox_135_tls() {
    let tls = firefox_135().tls;
    assert_eq!(
        tls.ciphers,
        "TLS_AES_128_GCM_SHA256:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_256_GCM_SHA384:\
         TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:\
         TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:\
         TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:\
         TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA:TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA:\
         TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:\
         TLS_RSA_WITH_AES_128_GCM_SHA256:TLS_RSA_WITH_AES_256_GCM_SHA384:\
         TLS_RSA_WITH_AES_128_CBC_SHA:TLS_RSA_WITH_AES_256_CBC_SHA"
    );
    assert_eq!(
        tls.curves,
        "X25519MLKEM768:X25519:P-256:P-384:P-521:ffdhe2048:ffdhe3072"
    );
    assert_eq!(
        tls.sigalgs,
        "ecdsa_secp256r1_sha256:ecdsa_secp384r1_sha384:ecdsa_secp521r1_sha512:\
         rsa_pss_rsae_sha256:rsa_pss_rsae_sha384:rsa_pss_rsae_sha512:\
         rsa_pkcs1_sha256:rsa_pkcs1_sha384:rsa_pkcs1_sha512:ecdsa_sha1:rsa_pkcs1_sha1"
    );
    assert_eq!(tls.alpn, &["h2", "http/1.1"]);
    assert!(!tls.grease);
    assert!(!tls.permute_extensions);
    assert!(tls.ech_grease);
    assert!(tls.pre_shared_key);
    assert!(tls.psk_skip_session_tickets);
    assert!(tls.psk_dhe_ke);
    assert!(tls.session_ticket);
    assert!(tls.ocsp_stapling);
    assert!(tls.signed_cert_timestamps);
    assert_eq!(tls.cert_compressors, "zlib,brotli,zstd");
    assert_eq!(tls.min_tls, TlsVersion::Tls12);
    assert_eq!(tls.max_tls, TlsVersion::Tls13);
    assert_eq!(tls.key_shares, &["X25519MLKEM768", "X25519", "P-256"]);
    assert_eq!(
        tls.delegated_credentials,
        "ecdsa_secp256r1_sha256:ecdsa_secp384r1_sha384:ecdsa_secp521r1_sha512:ecdsa_sha1"
    );
    assert_eq!(tls.record_size_limit, 0x4001);
    assert_eq!(
        tls.extension_order,
        &[
            0, 23, 65281, 10, 11, 35, 16, 5, 34, 18, 51, 43, 13, 45, 28, 27, 65037
        ]
    );
}

#[test]
fn firefox_135_http2() {
    let h2 = firefox_135().http2;
    assert_eq!(h2.initial_stream_id, Some(3));
    assert_eq!(h2.initial_window_size, Some(131072));
    assert_eq!(h2.initial_connection_window_size, Some(12582912));
    assert_eq!(h2.max_frame_size, Some(16384));
    assert_eq!(h2.header_table_size, Some(65536));
    assert_eq!(h2.enable_push, Some(false));
    assert_eq!(h2.max_header_list_size, None);
    assert_eq!(h2.max_concurrent_streams, None);
    let dep = h2.stream_dependency.expect("firefox dep");
    assert_eq!((dep.parent, dep.weight, dep.exclusive), (0, 41, false));
    assert!(h2.priorities.is_empty());
    assert_eq!(h2.pseudo_order, &["method", "path", "authority", "scheme"]);
}

#[test]
fn firefox_135_headers_and_ua() {
    let p = firefox_135();
    assert_eq!(p.header_set, HeaderSet::FirefoxZstd);
    assert_eq!(p.sec_ch_ua(), None);
    assert_eq!(
        p.user_agent(Platform::MacOS).as_deref(),
        Some(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0"
        )
    );
    assert_eq!(
        p.user_agent(Platform::Windows).as_deref(),
        Some("Mozilla/5.0 (Windows NT 10.0; rv:135.0) Gecko/20100101 Firefox/135.0")
    );
    let tpl = header_template(p.header_set);
    assert_eq!(tpl.first().map(|f| f.name), Some("user-agent"));
    assert_eq!(tpl.last().map(|f| f.name), Some("te"));
    let names: Vec<&str> = tpl.iter().map(|f| f.name).collect();
    assert!(names.contains(&"priority"));
    // sec-ch-ua fields must not resolve on a Firefox profile.
    let chrome_tpl = header_template(HeaderSet::Chrome);
    let sec = chrome_tpl.iter().find(|f| f.name == "sec-ch-ua").unwrap();
    assert_eq!(p.header_value(Platform::MacOS, sec), None);
}

fn safari_18() -> &'static Profile {
    profile("safari_18").expect("safari_18 must exist")
}

#[test]
fn safari_18_tls() {
    let tls = safari_18().tls;
    assert_eq!(
        tls.ciphers,
        "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:\
         TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:\
         TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:\
         TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:\
         TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA:TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA:\
         TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:\
         TLS_RSA_WITH_AES_256_GCM_SHA384:TLS_RSA_WITH_AES_128_GCM_SHA256:\
         TLS_RSA_WITH_AES_256_CBC_SHA:TLS_RSA_WITH_AES_128_CBC_SHA:\
         TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA:TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA:\
         TLS_RSA_WITH_3DES_EDE_CBC_SHA"
    );
    assert_eq!(tls.curves, "X25519:P-256:P-384:P-521");
    assert_eq!(
        tls.sigalgs,
        "ecdsa_secp256r1_sha256:rsa_pss_rsae_sha256:rsa_pkcs1_sha256:\
         ecdsa_secp384r1_sha384:ecdsa_sha1:rsa_pss_rsae_sha384:rsa_pss_rsae_sha384:\
         rsa_pkcs1_sha384:rsa_pss_rsae_sha512:rsa_pkcs1_sha512:rsa_pkcs1_sha1"
    );
    assert!(tls.grease);
    assert!(!tls.permute_extensions);
    assert!(!tls.ech_grease);
    assert!(!tls.session_ticket);
    assert!(tls.ocsp_stapling);
    assert!(tls.signed_cert_timestamps);
    assert_eq!(tls.cert_compressors, "zlib");
    assert_eq!(tls.min_tls, TlsVersion::Tls10);
    assert_eq!(tls.max_tls, TlsVersion::Tls13);
    assert!(tls.alpn.is_empty());
    assert!(tls.key_shares.is_empty());
}

#[test]
fn safari_18_http2() {
    let h2 = safari_18().http2;
    assert_eq!(h2.max_concurrent_streams, Some(100));
    assert_eq!(h2.initial_window_size, Some(2097152));
    assert_eq!(h2.initial_connection_window_size, Some(10485760));
    assert_eq!(h2.enable_push, Some(false));
    assert!(h2.enable_connect_protocol);
    assert!(h2.no_rfc7540_priorities);
    assert_eq!(h2.header_table_size, None);
    assert_eq!(h2.stream_dependency, None);
    assert_eq!(h2.pseudo_order, &["method", "scheme", "authority", "path"]);
}

#[test]
fn safari_18_headers_and_ua() {
    let p = safari_18();
    assert_eq!(p.header_set, HeaderSet::Safari18);
    assert_eq!(p.sec_ch_ua(), None);
    assert_eq!(
        p.user_agent(Platform::MacOS).as_deref(),
        Some(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
             (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
        )
    );
    // safari_18 itself ships only the macOS UA; iOS UAs are separate profiles.
    assert_eq!(p.user_agent(Platform::Ios), None);
    let ios = profile("safari_ios_18.1.1").expect("safari_ios_18.1.1 must exist");
    assert_eq!(ios.header_set, HeaderSet::Safari18);
    assert_eq!(
        ios.user_agent(Platform::Ios).as_deref(),
        Some(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1_1 like Mac OS X) AppleWebKit/605.1.15 \
             (KHTML, like Gecko) Version/18.1.1 Mobile/15E148 Safari/604.1"
        )
    );
    let tpl = header_template(p.header_set);
    let names: Vec<&str> = tpl.iter().map(|f| f.name).collect();
    assert_eq!(
        names,
        [
            "sec-fetch-dest",
            "user-agent",
            "accept",
            "sec-fetch-site",
            "sec-fetch-mode",
            "accept-language",
            "priority",
            "accept-encoding"
        ]
    );
    let dest = &tpl[0];
    assert_eq!(
        p.header_value(Platform::MacOS, dest).as_deref(),
        Some("document")
    );
}

#[test]
fn registry_has_all_137_profiles() {
    let all = profiles();
    assert_eq!(all.len(), 137);
    for p in all {
        assert_eq!(profile(p.name), Some(p));
        assert!(!p.tls.ciphers.is_empty(), "{}", p.name);
        assert!(!p.http2.pseudo_order.is_empty(), "{}", p.name);
        assert!(
            Platform::ALL.iter().any(|pl| p.user_agent(*pl).is_some()),
            "{}",
            p.name
        );
        let tpl = header_template(p.header_set);
        assert!(!tpl.is_empty(), "{}", p.name);
        // Every template's user-agent slot must resolve on some platform.
        let ua_field = tpl
            .iter()
            .find(|f| matches!(f.value, HeaderValue::UserAgent));
        if let Some(field) = ua_field {
            assert!(
                Platform::ALL
                    .iter()
                    .any(|pl| p.header_value(*pl, field).is_some()),
                "{}",
                p.name
            );
        }
    }
}

#[test]
fn unknown_profile_name_returns_none() {
    assert_eq!(profile("chrome_999"), None);
    assert_eq!(profile(""), None);
    assert_eq!(profile("Chrome_131"), None);
}

#[test]
fn ua_pool_decodes_to_nonempty_strings() {
    for i in 0..data::UA_POOL.len() {
        let s = super::ua_pool_get(i as u16);
        assert!(!s.is_empty(), "pool entry {i}");
    }
}
