# VTU alan dışa aktarma: ön kayıt (W1–W6)

Ön kayıt 2026-11-26 02:42 +0300, yazıcı yazılmadan ve hiçbir koşudan önce (08 §4: alan dışa aktarma, eşleme,
tensör sözleşmesi, roundtrip).

**Ne.** `toporlogy.execution.vtu`: düzenli ızgaranın **etkin** elemanları VTK XML UnstructuredGrid (`.vtu`, ASCII,
float64 `repr` ile tam hassasiyet) olarak; nokta verisi (yer değiştirme, yük durumu başına), hücre verisi (x̄, E,
solid von Mises merkez gerilmesi), alan verisi (tensör/Voigt sözleşmesi, birimler, kaynak SHA). Dış bağımlılık yok;
vtk yalnız testte bağımsız okuyucu olarak (yoksa ilgili test NOT VERIFIED olarak atlanır). Post adımı `vtu`.

Eleman düğüm sırası (`Domain.edof[:, ::dim] // dim`) VTK_QUAD (9) ve VTK_HEXAHEDRON (12) sırasıyla aynı (yazmadan
önce `Domain((2, 1))` ve `Domain((1, 1, 1))` koordinatlarından okundu).

| # | Kontrol | Eşik |
|:---|:---|:---|
| W1 | Kendi okuyucuyla roundtrip: koordinatlar, bağlantı, hücre tipi, her dizi | bit düzeyinde aynı |
| W2 | vtk (9.x) `vtkXMLUnstructuredGridReader` aynı dizileri, hücre sayısını ve tipini okur | bit düzeyinde aynı |
| W3 | vtk `vtkCellSizeFilter`: her hücrenin alanı/hacmi > 0 ve toplam = n_active · h^dim | ≤ 1e-12 göreli |
| W4 | Delikli (void bölgeli) ve kısmi hacimli bir alan: yalnız etkin elemanlar yazılır, hücre sırası `domain.active_elements` | aynen |
| W5 | Yanlış şekilli dizi (ne düğüm ne hücre sayısında), NaN içeren dizi değil (NaN yazılır), bilinmeyen sözleşme | açık `ValueError`; NaN roundtrip'te korunur |
| W6 | `vtu` post adımı: bir örneğin koşusu `result.vtu` yazar ve W1 okuyucusu x̄'yi `result.npz`'deki `xphys` ile bit düzeyinde aynı okur | aynen |
