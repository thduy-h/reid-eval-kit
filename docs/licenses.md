# License review

The repository's original implementation and independent test oracle are MIT
licensed. No Torchreid or FastReID source code is copied or vendored; their
public functions were read to verify protocol behavior. The inspected
[Torchreid license](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE)
is MIT and [FastReID's](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE)
is Apache-2.0; reading algorithms is not redistribution of these packages.
Do not copy code later without preserving applicable notices and obligations.

The locally used NumPy 2.4.4 metadata declares
`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`; its binary wheels bundle
additional licensed components (notably OpenBLAS and Fortran runtime libraries)
with notices in their own distribution. These permissive notices do not require relicensing this toolkit. NumPy is a declared dependency, not
bundled into the reid-eval-kit wheel. The optional PyYAML reader declares MIT.
The built project wheel contains only toolkit source, metadata, and its MIT
license. Development tools are not mandatory runtime dependencies. Dependency
licenses and source-package metadata were inspected during the local audit;
this is an engineering compatibility review, not a legal opinion about future
unreviewed dependencies or versions. No dataset assets are redistributed.
