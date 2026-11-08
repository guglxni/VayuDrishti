import type { ExpressionSpecification, StyleSpecification } from 'maplibre-gl'

/**
 * Key-free basemap: OpenFreeMap vector tiles (OpenMapTiles schema) restyled to the
 * VayuDrishti ink palette, plus AWS Terrarium elevation for hillshade.
 * Land is lifted slightly above the sea and the shoreline carries a faint lit edge,
 * so Mumbai's peninsula reads clearly underneath the data layers.
 */
const C = {
  sea: '#08111a',
  land: '#141b22',
  park: '#14201d',
  shore: '#3a5367',
  minor: '#1c242c',
  major: '#29323b',
  motorway: '#3a4652',
  rail: '#2c3640',
  runway: '#1f272f',
  building: '#1b232c',
  label: '#8893a0',
  labelDim: '#5d6874',
  water: '#3f5a70',
  halo: '#0a1119',
}

const lineString: ExpressionSpecification = ['match', ['geometry-type'], ['LineString', 'MultiLineString'], true, false]
const polygon: ExpressionSpecification = ['match', ['geometry-type'], ['Polygon', 'MultiPolygon'], true, false]
const nameField: ExpressionSpecification = ['coalesce', ['get', 'name:en'], ['get', 'name_en'], ['get', 'name']]

export const BEFORE_DATA_LAYER = 'building-3d'
export const BEFORE_LABELS = 'label-neighbourhood'

export function buildBasemapStyle(): StyleSpecification {
  return {
    version: 8,
    glyphs: 'https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf',
    sources: {
      omt: {
        type: 'vector',
        url: 'https://tiles.openfreemap.org/planet',
        attribution:
          '<a href="https://openfreemap.org" target="_blank">OpenFreeMap</a> © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>',
      },
      dem: {
        type: 'raster-dem',
        tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
        encoding: 'terrarium',
        tileSize: 256,
        maxzoom: 14,
        attribution: 'Terrain: <a href="https://registry.opendata.aws/terrain-tiles/" target="_blank">Mapzen / AWS</a>',
      },
    },
    layers: [
      { id: 'background', type: 'background', paint: { 'background-color': C.land } },
      {
        id: 'hillshade',
        type: 'hillshade',
        source: 'dem',
        paint: {
          'hillshade-exaggeration': 0.45,
          'hillshade-shadow-color': '#05080c',
          'hillshade-highlight-color': '#2c3742',
          'hillshade-accent-color': '#0d1318',
          'hillshade-illumination-direction': 315,
        },
      },
      {
        id: 'landcover-wood',
        type: 'fill',
        source: 'omt',
        'source-layer': 'landcover',
        filter: ['all', polygon, ['match', ['get', 'class'], ['wood', 'grass', 'wetland'], true, false]],
        paint: { 'fill-color': C.park, 'fill-opacity': 0.75 },
      },
      {
        id: 'landuse-park',
        type: 'fill',
        source: 'omt',
        'source-layer': 'landuse',
        filter: ['all', polygon, ['match', ['get', 'class'], ['park', 'cemetery', 'grass'], true, false]],
        paint: { 'fill-color': C.park, 'fill-opacity': 0.6 },
      },
      {
        id: 'water',
        type: 'fill',
        source: 'omt',
        'source-layer': 'water',
        filter: ['all', polygon, ['!=', ['get', 'brunnel'], 'tunnel']],
        paint: { 'fill-color': C.sea, 'fill-antialias': true },
      },
      {
        id: 'shoreline-glow',
        type: 'line',
        source: 'omt',
        'source-layer': 'water',
        filter: ['all', polygon, ['!=', ['get', 'brunnel'], 'tunnel']],
        paint: {
          'line-color': C.shore,
          'line-width': ['interpolate', ['linear'], ['zoom'], 9, 2.5, 13, 5],
          'line-blur': ['interpolate', ['linear'], ['zoom'], 9, 2.5, 13, 5],
          'line-opacity': 0.28,
        },
      },
      {
        id: 'shoreline',
        type: 'line',
        source: 'omt',
        'source-layer': 'water',
        filter: ['all', polygon, ['!=', ['get', 'brunnel'], 'tunnel']],
        paint: {
          'line-color': C.shore,
          'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.6, 14, 1.2],
          'line-opacity': 0.75,
        },
      },
      {
        id: 'waterway',
        type: 'line',
        source: 'omt',
        'source-layer': 'waterway',
        filter: lineString,
        paint: { 'line-color': C.sea, 'line-width': ['interpolate', ['linear'], ['zoom'], 10, 0.6, 15, 2.5] },
      },
      {
        id: 'runway',
        type: 'line',
        source: 'omt',
        'source-layer': 'aeroway',
        minzoom: 10,
        filter: ['all', lineString, ['match', ['get', 'class'], ['runway', 'taxiway'], true, false]],
        paint: {
          'line-color': C.runway,
          'line-width': ['interpolate', ['exponential', 1.5], ['zoom'], 10, 2, 16, 40],
        },
      },
      {
        id: 'road-minor',
        type: 'line',
        source: 'omt',
        'source-layer': 'transportation',
        minzoom: 12,
        filter: ['all', lineString, ['match', ['get', 'class'], ['minor', 'service'], true, false]],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': C.minor,
          'line-width': ['interpolate', ['exponential', 1.5], ['zoom'], 12, 0.4, 16, 5],
        },
      },
      {
        id: 'road-major',
        type: 'line',
        source: 'omt',
        'source-layer': 'transportation',
        minzoom: 9,
        filter: ['all', lineString, ['match', ['get', 'class'], ['primary', 'secondary', 'tertiary'], true, false]],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': C.major,
          'line-width': ['interpolate', ['exponential', 1.4], ['zoom'], 9, 0.4, 12, 1.2, 16, 8],
        },
      },
      {
        id: 'rail',
        type: 'line',
        source: 'omt',
        'source-layer': 'transportation',
        minzoom: 10,
        filter: ['all', lineString, ['==', ['get', 'class'], 'rail'], ['!', ['has', 'service']]],
        paint: {
          'line-color': C.rail,
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 0.6, 15, 1.6],
          'line-dasharray': [3, 2],
        },
      },
      {
        id: 'road-motorway',
        type: 'line',
        source: 'omt',
        'source-layer': 'transportation',
        minzoom: 7,
        filter: ['all', lineString, ['match', ['get', 'class'], ['motorway', 'trunk'], true, false]],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': C.motorway,
          'line-width': ['interpolate', ['exponential', 1.4], ['zoom'], 8, 0.6, 12, 1.8, 16, 12],
        },
      },
      {
        id: BEFORE_DATA_LAYER,
        type: 'fill-extrusion',
        source: 'omt',
        'source-layer': 'building',
        minzoom: 12.5,
        filter: ['!', ['has', 'hide_3d']],
        paint: {
          'fill-extrusion-color': [
            'interpolate',
            ['linear'],
            ['coalesce', ['get', 'render_height'], 6],
            0,
            C.building,
            60,
            '#222c36',
            200,
            '#2b3742',
          ],
          'fill-extrusion-height': ['interpolate', ['linear'], ['zoom'], 12.5, 0, 13.5, ['coalesce', ['get', 'render_height'], 6]],
          'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
          'fill-extrusion-opacity': 0.92,
          'fill-extrusion-vertical-gradient': true,
        },
      },
      {
        id: 'water-name',
        type: 'symbol',
        source: 'omt',
        'source-layer': 'water_name',
        filter: ['match', ['geometry-type'], ['Point', 'MultiPoint'], true, false],
        layout: {
          'text-field': nameField,
          'text-font': ['Noto Sans Italic'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 8, 11, 12, 14],
          'text-letter-spacing': 0.25,
          'text-max-width': 6,
        },
        paint: { 'text-color': C.water, 'text-halo-color': C.halo, 'text-halo-width': 1 },
      },
      {
        id: BEFORE_LABELS,
        type: 'symbol',
        source: 'omt',
        'source-layer': 'place',
        minzoom: 10,
        maxzoom: 16,
        filter: ['match', ['get', 'class'], ['suburb', 'neighbourhood', 'quarter'], true, false],
        layout: {
          'text-field': nameField,
          'text-font': ['Noto Sans Regular'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 10, 10, 14, 13],
          'text-letter-spacing': 0.04,
          'text-max-width': 8,
          'text-padding': 6,
        },
        paint: {
          'text-color': ['interpolate', ['linear'], ['zoom'], 10, C.labelDim, 13, C.label],
          'text-halo-color': C.halo,
          'text-halo-width': 1.2,
        },
      },
      {
        id: 'label-city',
        type: 'symbol',
        source: 'omt',
        'source-layer': 'place',
        maxzoom: 11.5,
        filter: ['match', ['get', 'class'], ['city', 'town'], true, false],
        layout: {
          'text-field': nameField,
          'text-font': ['Noto Sans Bold'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 8, 12, 11, 15],
          'text-letter-spacing': 0.2,
          'text-transform': 'uppercase',
        },
        paint: { 'text-color': '#a3adb8', 'text-halo-color': C.halo, 'text-halo-width': 1.5, 'text-opacity': 0.8 },
      },
    ],
  }
}
