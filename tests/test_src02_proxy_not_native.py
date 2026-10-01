"""SRC-2: a multiscale rolling-filter proxy cannot certify native-wavelet coverage."""
import unittest

from tests._src_fixtures import L


class ProxyCannotCertifyNative(unittest.TestCase):
    def test_proxy_materialized_does_not_cover_native_wavelet(self):
        row = L.transform_row("ohlc_price_bar", "wavelet", method_id="PROXY_ROLLING_MEAN_MULTISCALE_16_32_64_128",
                              states={"implemented": True, "materialized": True})
        self.assertFalse(L.certifies(row, "wavelet_native_dwt"))
        self.assertTrue(L.certifies(row, "wavelet_proxy_multiscale"))
        cov = L.family_coverage([row])
        self.assertEqual(cov["wavelet_native_dwt"]["state"], "NOT_COVERED")
        self.assertIn("PROXY", cov["wavelet_native_dwt"]["reason"])

    def test_native_requires_a_native_method_id(self):
        row = L.transform_row("ohlc_price_bar", "wavelet", method_id="NATIVE_DWT_db4_pywavelets",
                              states={"implemented": True})
        self.assertTrue(L.certifies(row, "wavelet_native_dwt"))

    def test_not_applicable_needs_domain_justification(self):
        with self.assertRaises(ValueError):
            L.transform_row("single_macro_value", "technical_ohlc", method_id="NA", states={}, not_applicable="")
        r = L.transform_row("single_macro_value", "technical_ohlc", method_id="NA", states={},
                            not_applicable="an OHLC indicator needs open/high/low/close; a macro release is one value")
        self.assertEqual(r["state"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
