import React from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Tooltip,
  Box,
  Heading,
} from '@chakra-ui/react';

export function TrainingConfig({ config, onChange }) {
  const renderNumberInput = (category, key, params) => (
    <FormControl key={key}>
      <Tooltip label={params.description}>
        <FormLabel>{key}</FormLabel>
      </Tooltip>
      <NumberInput
        min={params.min}
        max={params.max}
        defaultValue={params.default}
        onChange={(value) => onChange(category, key, parseFloat(value))}
        step={(params.max - params.min) / 100}
      >
        <NumberInputField />
        <NumberInputStepper>
          <NumberIncrementStepper />
          <NumberDecrementStepper />
        </NumberInputStepper>
      </NumberInput>
    </FormControl>
  );

  return (
    <VStack spacing={6} align="stretch">
      {Object.entries(config).map(([category, params]) => (
        <Box key={category} p={4} borderWidth={1} borderRadius="md">
          <Heading size="md" mb={4}>{category.replace(/_/g, ' ').toUpperCase()}</Heading>
          <VStack spacing={4} align="stretch">
            {Object.entries(params).map(([key, value]) =>
              renderNumberInput(category, key, value)
            )}
          </VStack>
        </Box>
      ))}
    </VStack>
  );
}