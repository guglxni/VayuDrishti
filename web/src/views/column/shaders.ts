// Colours passed to these ShaderMaterials are raw sRGB (no colour-space chunk), so PM2.5 hues match pmColor exactly.

export const PARTICLE_VERT = /* glsl */ `
uniform float uTime;
uniform float uBase;
uniform float uTop;
uniform float uFrac;
uniform float uSize;
uniform float uPx;
uniform float uMixPhase;
uniform float uStrat;
uniform float uR;
uniform float uWobble;
uniform vec2 uDrift;
attribute vec4 aRand;
attribute vec2 aMeta;
varying float vAlpha;
varying float vDepth;

float tri(float x) { return 1.0 - abs(mod(x, 2.0) - 1.0); }

void main() {
  float ph = aRand.w * 6.28318;
  vec2 xz = aRand.xy * uR + uDrift * (0.8 + 0.4 * aRand.w);
  xz += vec2(sin(uTime * 0.37 + ph * 3.0), cos(uTime * 0.29 + ph * 5.0)) * uWobble;
  xz = mod(xz + uR, 2.0 * uR) - uR;
  float r = length(xz);
  float yn = tri(aRand.z * 2.0 + uMixPhase * (0.45 + aRand.w));
  yn = pow(yn, uStrat);
  float y = mix(uBase, uTop, yn);
  vec4 mv = modelViewMatrix * vec4(xz.x, y, xz.y, 1.0);
  gl_Position = projectionMatrix * mv;
  float vis = 1.0 - smoothstep(uFrac - 0.04, uFrac, aMeta.x);
  float edge = 1.0 - smoothstep(uR * 0.84, uR, r);
  vAlpha = vis * edge;
  vDepth = yn;
  gl_PointSize = vAlpha < 0.002 ? 0.0 : uSize * aMeta.y * uPx * (40.0 / -mv.z);
}
`

export const PARTICLE_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vAlpha;
varying float vDepth;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = dot(c, c) * 4.0;
  if (d > 1.0) discard;
  float a = (1.0 - d);
  a *= a;
  vec3 col = uColor * (0.88 + 0.12 * vDepth) + (1.0 - d) * 0.08;
  gl_FragColor = vec4(col, a * vAlpha * uOpacity);
}
`

export const STREAK_VERT = /* glsl */ `
uniform vec2 uDir;
uniform float uTravel;
uniform float uLen;
uniform float uLid;
uniform float uR;
uniform float uWidth;
attribute vec2 corner;
attribute vec3 aInst;
varying float vS;
varying float vSide;
varying float vFade;

void main() {
  vec3 dir = vec3(uDir.x, 0.0, uDir.y);
  vec3 perp = vec3(-uDir.y, 0.0, uDir.x);
  float span = 2.0 * uR + uLen;
  float speedJ = 0.85 + 0.3 * fract(aInst.z * 7.31);
  float along = mod(uTravel * speedJ + aInst.z * span, span) - uR;
  float lane = aInst.x * uR;
  float pa = along - uLen * (1.0 - corner.x);
  vec3 p = perp * lane + dir * pa;
  p.y = max(0.22, aInst.y * uLid);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vec3 dv = normalize((modelViewMatrix * vec4(dir, 0.0)).xyz + vec3(1e-5));
  vec3 side = cross(dv, normalize(-mv.xyz));
  float sl = length(side);
  side = sl > 1e-4 ? side / sl : vec3(0.0, 1.0, 0.0);
  mv.xyz += side * corner.y * uWidth;
  gl_Position = projectionMatrix * mv;
  vS = corner.x;
  vSide = corner.y;
  float lim = sqrt(max(uR * uR - lane * lane, 0.0));
  vFade = 1.0 - smoothstep(lim * 0.55, lim, abs(pa));
}
`

export const STREAK_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vS;
varying float vSide;
varying float vFade;
void main() {
  float across = 1.0 - vSide * vSide;
  float a = pow(vS, 1.6) * across;
  a += smoothstep(0.9, 1.0, vS) * 0.35 * across;
  gl_FragColor = vec4(uColor, a * vFade * uOpacity);
}
`

/** Shared vertex shader for the column shell and haze (cylinder with y in [0, 1]). */
export const COLUMN_VERT = /* glsl */ `
varying float vY;
varying float vRim;
void main() {
  vY = position.y;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  vec3 v = normalize(-mv.xyz);
  vRim = 1.0 - abs(dot(n, v));
  gl_Position = projectionMatrix * mv;
}
`

export const SHELL_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vY;
varying float vRim;
void main() {
  float a = 0.02 + 0.42 * pow(vRim, 6.0);
  a *= 0.55 + 0.45 * smoothstep(0.0, 1.0, vY);
  gl_FragColor = vec4(uColor, a * uOpacity);
}
`

export const HAZE_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uDensity;
varying float vY;
varying float vRim;
void main() {
  float body = pow(1.0 - vRim, 0.8);
  float a = uDensity * body * (1.0 - 0.6 * vY) * smoothstep(1.0, 0.92, vY);
  gl_FragColor = vec4(uColor, a);
}
`

/** Flat disc shaders: vP is the local xy of a CircleGeometry. */
export const DISC_VERT = /* glsl */ `
varying vec2 vP;
void main() {
  vP = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

export const LID_FRAG = /* glsl */ `
uniform float uTime;
uniform float uR;
uniform float uOuter;
uniform vec3 uColor;
uniform float uOpacity;
varying vec2 vP;
void main() {
  float r = length(vP);
  float inside = 1.0 - smoothstep(uR - 0.08, uR + 0.08, r);
  float fade = 1.0 - smoothstep(uR, uOuter, r);
  float wave = 0.5 + 0.5 * sin(r * 2.6 - uTime * 1.1);
  float cell = sin(vP.x * 1.9 + uTime * 0.5) * sin(vP.y * 1.6 - uTime * 0.4);
  float shimmer = 0.7 + 0.3 * wave + 0.12 * cell;
  float a = (inside * 0.17 + (1.0 - inside) * 0.075 * fade) * shimmer;
  gl_FragColor = vec4(uColor, a * uOpacity);
}
`

export const FOOT_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uR;
varying vec2 vP;
void main() {
  float r = length(vP) / uR;
  float a = (1.0 - smoothstep(0.0, 1.0, r)) * 0.55 + smoothstep(0.9, 1.0, r) * 0.25;
  gl_FragColor = vec4(uColor, a * uOpacity);
}
`

export const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = p.xyww;
}
`

export const SKY_FRAG = /* glsl */ `
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform float uSun;
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  vec3 col = mix(uHorizon, uZenith, smoothstep(0.0, 0.55, h));
  col = mix(col, uHorizon * 0.82, smoothstep(0.0, -0.25, h));
  float s = max(dot(d, normalize(uSunDir)), 0.0);
  col += uSunCol * (pow(s, 900.0) * 0.6 + pow(s, 24.0) * 0.12 + pow(s, 4.0) * 0.05) * uSun;
  gl_FragColor = vec4(col, 1.0);
}
`
