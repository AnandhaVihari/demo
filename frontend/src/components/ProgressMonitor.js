import React from 'react';
import {
  Box,
  Progress,
  Text,
  VStack,
} from '@chakra-ui/react';

export function ProgressMonitor() {
  return (
    <VStack spacing={4} align="stretch">
      <Text fontWeight="bold">Training Progress</Text>
      <Progress size="sm" isIndeterminate />
      <Box>
        <Text fontSize="sm" color="gray.600">
          Training is in progress. This may take several minutes or hours depending on your dataset size and configuration.
        </Text>
      </Box>
    </VStack>
  );
}